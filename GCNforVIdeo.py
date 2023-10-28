import torch
import torch.nn as nn
import mediapipe as mp
import random
import os
import numpy as np
import wandb
from sklearn.metrics import f1_score,precision_score,recall_score
from loader.dataloader import MultiModalData
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from util.log import Log

# wandb.login(
#     key='0c9e631211d579a62fa94880e8e3efcf09cd66a0',
# )

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="Multimodal-HAR",
#     entity='aiotlab',
#     group='GCN_for_Video',
#     name=f'GCN_for_Video_lrate:{1e-3} - epoch:{60}',
#     # Track hyperparameters and run metadata
#     config={
#         "epochs": 60,
#         'random_seed': 20,
#         "n_classes": 41,
#         "batch_size": 128,
#         "device":'cuda:1'
#     })


def find_adjacency_matrix():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    adj = torch.zeros((21, 21))
    for connection in mp_hands.HAND_CONNECTIONS:
        adj[connection[0], connection[1]] = 1
        adj[connection[1], connection[0]] = 1
    return adj


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_vetex, device, dropout=0.5, bias=True):
        super(GraphConvolution, self).__init__()

        self.alpha = 1.
        self.device = device
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(
            input_dim, output_dim)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim)).to(device)
        else:
            self.bias = None

        for w in [self.weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()

        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)

        return m_norm

    def forward(self, adj, x):

        x = self.dropout(x)

        # K-ordered Chebyshev polynomial
        adj_norm = self.normalize(adj)
        sqr_norm = self.normalize(torch.mm(adj, adj))
        m_norm = (self.alpha*adj_norm + (1.-self.alpha)
                  * sqr_norm).to(self.device)

        x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
        x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
        if self.bias is not None:
            x_out += self.bias
        x_out = self.act(x_out)

        return x_out


class StandConvolution(nn.Module):
    def __init__(self, dims, num_classes, dropout, device):
        super(StandConvolution, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3,
                      stride=(1, 2), padding=1),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=3,
                      stride=(1, 2), padding=1),
            nn.InstanceNorm2d(
                dims[2]),
            nn.ReLU(
                inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[2], dims[3], kernel_size=3,
                      stride=(1, 2), padding=1),
            nn.InstanceNorm2d(
                dims[3]),
            nn.ReLU(
                inplace=True),
            # nn.AvgPool2d(3, stride=2)
        ).to(device)

        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)

        # x_out = self.fc(x_tmp.view(x.size(0), -1)) # multimodal
        x_out = x_tmp.view(x.size(0), -1)

        return x_out


class GGCN(nn.Module):
    def __init__(self, adj, num_classes, gc_dims, sc_dims, device, dropout=0.2):
        super(GGCN, self).__init__()

        adj = adj + torch.eye(adj.size(0)).to(adj).detach()
        ident = torch.eye(adj.size(0)).to(adj)
        zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
        self.adj = torch.cat([torch.cat([adj, adj, zeros], 1),
                              torch.cat([adj, adj, adj], 1),
                              torch.cat([zeros, adj, adj], 1)], 0).float()

        self.gcl = GraphConvolution(
            gc_dims[0], gc_dims[1], 21, dropout=dropout, device=device)
        self.conv = StandConvolution(
            sc_dims, num_classes, dropout=dropout, device=device)

    def forward(self, x):
        x = torch.cat([x[:, :-2], x[:, 1:-1], x[:, 2:]], dim=2)
        multi_conv = self.gcl(self.adj, x)
        logit = self.conv(multi_conv) 
        return logit
    

def train(train_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()
    for videos, labels, emgs in tqdm(train_loader):
       
        

        videos = videos.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(videos)
        loss = criterion(outputs,labels)
            
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    epoch_loss = running_loss / (len(train_loader))
   

    return model, epoch_loss, optimizer

def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0
    

    for videos, labels, emgs in tqdm(valid_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(videos)
        loss = criterion(outputs,labels)
        
        running_loss += loss.item()

    epoch_loss = running_loss / (len(valid_loader))

    epoch_loss = running_loss / (len(valid_loader))
   
    return model, epoch_loss

def get_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    predicted_labels = []
    truth_labels = []

    model.eval()
    with torch.no_grad():
        for videos, labels, emgs in data_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            emgs = emgs.to(device).double()

            # forward
            outputs = model(videos)
           
            predicted = torch.argmax(torch.softmax(outputs, 1), 1)
          
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted)
            truth_labels.extend(labels)

    f1_micro = f1_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='micro')
    precision_score_f1 = precision_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='micro')
    recall_score_f1 = recall_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='micro')
    
    return correct/total, f1_micro, precision_score_f1,recall_score_f1


def trainer():

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    seed_everything(run.config['random_seed'])

    trainset = MultiModalData("data/new_data/new_train_files.pkl")
    testset = MultiModalData("data/new_data/new_test_files.pkl")
    valset = MultiModalData("data/new_data/new_val_files.pkl")

    train_loader = DataLoader(trainset, batch_size=run.config['batch_size'],
                            drop_last=True, num_workers=0, shuffle=True)
    valid_loader = DataLoader(valset, batch_size=run.config['batch_size'],
                            drop_last=True, num_workers=0 )
    test_loader = DataLoader(testset, batch_size=run.config['batch_size'],
                            drop_last=True, num_workers=0)



    device = run.config['device']
    model =GGCN(find_adjacency_matrix(), 41,[3, 9], [9, 16, 32, 64], run.config["device"], 0.0).to(device)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())


    epochs = run.config['epochs']
    train_losses = []
    valid_losses = []
    train_accuracy = []
    val_accuracy = []

    train_score_log = [[], [], [], []]
    val_score_log = [[], [], [], [], []]
    test_score_log = [[], [], [], [], []]

    best_f1 = -1000

    log = Log("log/GCN", "gcn")


    for epoch in range(epochs):
        # training
        
        model, train_loss, optimizer = train(
                train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
    
        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, device)
        valid_losses.append(valid_loss)
        
        # get train score
        train_acc, f1_score_micro,precision_score_micro,recall_score_micro = get_accuracy(model, train_loader, device)
        train_score_log[0].append(train_acc)
        train_score_log[1].append(f1_score_micro)
        train_score_log[2].append(precision_score_micro)
        train_score_log[3].append(recall_score_micro)
        
        # get val score
        val_acc, f1_score_micro,precision_score_micro,recall_score_micro = get_accuracy(model, valid_loader, device)
        val_score_log[0].append(val_acc)
        val_score_log[1].append(f1_score_micro)
        val_score_log[2].append(precision_score_micro)
        val_score_log[3].append(recall_score_micro)
    
        # save f1 score
        if best_f1 < f1_score_micro:
            torch.save(model.state_dict(),
                    f"log/GCN/best_model{epoch}.pth")
            best_f1 = f1_score_micro
            
        # get test score
        test_acc, f1_score_micro,precision_score_micro,recall_score_micro = get_accuracy(model, test_loader, device)
        test_score_log[0].append(test_acc)
        test_score_log[1].append(f1_score_micro)
        test_score_log[2].append(precision_score_micro)
        test_score_log[3].append(recall_score_micro)
        
        wandb.log({
            "Train loss": wandb.plot.line_series(
                xs=range(len(train_losses)),
                ys=[train_losses],
                keys= ["Loss"],
                title="Train loss",
                xname="x epochs"
            ),
            "Val loss": wandb.plot.line_series(
                xs=range(len(valid_losses)),
                ys=[valid_losses],
                keys=["Loss"],
                title="Val loss",
                xname="x epochs"
            ),
            "Train Score": wandb.plot.line_series(
                xs=range(len(train_score_log[0])),
                ys=[train_score_log[0], train_score_log[1], train_score_log[2],train_score_log[3]],
                keys=["Accuracy", "F1_score_micro","precision_score_micro","recall_score_micro"],
                title="Train Score",
                xname="x epochs"),
            "Val Score": wandb.plot.line_series(
                xs=range(len(val_score_log[0])),
                ys=[val_score_log[0], val_score_log[1], val_score_log[2],val_score_log[3]],
                keys=["Accuracy", "F1_score_micro","precision_score_micro","recall_score_micro"],
                title="Val Score",
                xname="x epochs"),
            "Test Score": wandb.plot.line_series(
                xs=range(len(test_score_log[0])),
                ys=[test_score_log[0], test_score_log[1], test_score_log[2],test_score_log[3]],
                keys=["Accuracy", "F1_score_micro","precision_score_micro","recall_score_micro"],
                title="Test Score",
                xname="x epochs"),
        
        })
    wandb.run.finish()

