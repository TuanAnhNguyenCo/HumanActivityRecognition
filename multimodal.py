from trainer import ViT
from VITforEMGAndBone import ViTForEMGAndBone, CrossAttention
from GCNforVIdeo import GGCN, find_adjacency_matrix
from torch import nn
import torch
from einops import rearrange, repeat
import math
from loader.dataloader import MultiModalData
from torch.utils.data import Dataset, DataLoader
from util.log import Log
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm
import random
import os
import numpy as np
import wandb


wandb.login(
    key='9bce1a84793dd8652665e9c5a731d2f7775245ad',
    relogin=True
)

run = wandb.init(
    # Set the project where this run will be logged
    project="Missing_modality",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 100,
        'random_seed': 20,
        "common_dim": 64,
        "n_classes": 41,
        "batch_size": 128,
        "device":'cuda:0',
    })

class MultiModal(nn.Module):
    def __init__(self,common_dim,n_classes):
        super(MultiModal, self).__init__()
        self.gcn = GGCN(find_adjacency_matrix(), 41,
                        [3, 9], [9, 16, 32, 64], 0.0)
        self.vit = ViT(emg_size=(44100*0.2, 8), patch_height=60, num_classes=41, dim=126,
                       depth=5, mlp_dim=512, heads=8, pool='cls', dropout=0.35, emb_dropout=0.35).double()
        self.crossAtt1 = CrossAttention(63, 14112, 512).double()
        self.crossAtt2 = CrossAttention(14112, 63, 512).double()
        self.vitForEMGandBone = ViTForEMGAndBone(
            1024,  41, 512, 3, 8, 512, pool='cls', dim_head=64, dropout=0., emb_dropout=0.).double()
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Linear(common_dim,common_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.1)
        ).double()
        self.fc = nn.Linear(common_dim, n_classes).double()
        self.fc1 = nn.Linear(1536, common_dim).double()
        self.fc2 = nn.Linear(128, common_dim).double()
        self.fc3 = nn.Linear(2176, common_dim).double()

    def forward(self, bones, emg):

        x1 = self.gcn(bones)  # out
        x1 = self.drop(x1)
        x2 = self.vit(emg.double())  # out
        emg1 = rearrange(emg, "b (a d) c ->b a (d c) ", a=5).double()
        bone1 = rearrange(bones, "b t n c -> b t (n c)").double()
        x3 = self.crossAtt1(bone1, emg1)
        x4 = self.crossAtt2(emg1, bone1)
        x5 = self.vitForEMGandBone(torch.concat(
            [x3, x4], dim=-1).double())  # out
        x5 = torch.concat([x1, x2, x5], dim=-1)
        
        x5 = self.fc3(x5)
        
        x5 = self.classify(x5)
        
        return self.fc(x5)
        


def train(train_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()

    for videos, labels, emgs in tqdm(train_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()

        # forward
        outputs = model(videos, emgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / (len(train_loader))
    return model, epoch_loss, optimizer


def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for videos, labels, emgs in tqdm(valid_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()

        # forward

        outputs = model(videos, emgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    epoch_loss = running_loss / (len(valid_loader))
    return model, epoch_loss


def get_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    predicted_labels = []
    truth_labels = []

    model.eval()
    for videos, labels, emgs in data_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()

        # forward
        outputs = model(videos, emgs)
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

    return correct/total, f1_micro, precision_score_f1, recall_score_f1


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(run.config["random_seed"])

trainset = MultiModalData("data/new_data/new_train_files.pkl")
testset = MultiModalData("data/new_data/new_test_files.pkl")
valset = MultiModalData("data/new_data/new_val_files.pkl")

train_loader = DataLoader(trainset, batch_size=run.config['batch_size'],
                          drop_last=True, num_workers=3, shuffle=True)
valid_loader = DataLoader(valset, batch_size=run.config['batch_size'],
                          drop_last=True, num_workers=3 )
test_loader = DataLoader(testset, batch_size=run.config['batch_size'],
                         drop_last=True, num_workers=3)

device = run.config['device']

device = 'cuda:0'
model = MultiModal(
    common_dim=run.config['common_dim'], n_classes=run.config['n_classes']).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


epochs = 100
train_losses = []
valid_losses = []
train_accuracy = []
val_accuracy = []


train_score_log = [[], [], [], []]
val_score_log = [[], [], [], [], []]
test_score_log = [[], [], [], [], []]

best_f1 = -1000

log = Log("log/Multimodal", "multimodal")


for epoch in range(epochs):
    # training
    model, train_loss, optimizer = train(
        train_loader, model, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # validation
    with torch.no_grad():
        model, valid_loss = validate(valid_loader, model, criterion, device)
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
                   f"log/Multimodal/best_model{epoch}.pth")

    print("Epoch {} --- Train loss = {} --- Valid loss = {} -- Train set accuracy = {} % Valid set Accuracy = {} %".format
          (epoch+1, train_loss, valid_loss, train_acc, val_acc))
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


