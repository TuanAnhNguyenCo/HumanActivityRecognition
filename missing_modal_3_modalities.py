from trainer import ViT
from VIT_PSD import ViT as VIT_spec
from VITforEMGAndBone import ViTForEMGAndBone, CrossAttention
from GCNforVIdeo import GGCN, find_adjacency_matrix
from torch import nn
import torch
from einops import rearrange, repeat
import math
from loader.dataloader import ThreeModalData
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
        "epochs": 60,
        'random_seed': 20,
        "common_dim": 64,
        "n_classes": 41,
        "batch_size": 128,
        "T": 1,
        "device":'cuda:0',
        "cl_rate": 1,
        "param_vid": 1,
        "param_emg": 1,
        "param_spec": 1,
    })

class CrossAttentionFor2Modalites(nn.Module):
    def __init__(self,dim1,dim2,out_dim):
        super(CrossAttentionFor2Modalites, self).__init__()
        self.crossAtt1 = CrossAttention(dim1, dim2, out_dim).double()
        self.crossAtt2 = CrossAttention(dim2, dim1, out_dim).double()
        
    def forward(self,modality1,modality2):
        x1 =  self.crossAtt1(modality1,modality2)
        x2 =  self.crossAtt2(modality2,modality1)
        return torch.concat([x1,x2],dim = -1)

class MultiModal(nn.Module):
    def __init__(self, common_dim, n_classes):
        super(MultiModal, self).__init__()
        self.gcn = GGCN(find_adjacency_matrix(), 41,
                        [3, 9], [9, 16, 32, 64], run.config["device"], 0.0)
        self.vit = ViT(emg_size=(44100*0.2, 8), patch_height=60, num_classes=41, dim=128,
                       depth=5, mlp_dim=256, heads=8, pool='cls', dropout=0.35, emb_dropout=0.35).double()
        self.att1 = CrossAttentionFor2Modalites(63,14112,256).double()
        self.att2 = CrossAttentionFor2Modalites(63,14560,256).double()
        self.att3 = CrossAttentionFor2Modalites(14560,14112,256).double()
        self.vitForEMGandBone = ViTForEMGAndBone(
            1536,  41, 512, 3, 8, 512, pool='cls', dim_head=64, dropout=0, emb_dropout=0).double()
        self.spectrogram_model = VIT_spec(image_size=(130, 70), patch_size=(26, 14), num_classes=41, dim=128, depth=3,
            heads = 8, mlp_dim=256, pool='cls', channels=8, dim_head=64, dropout=0.3, emb_dropout=0.3, device=device
            ).to(device).double()

        self.fc = nn.Linear(common_dim, n_classes).double()
        # self.act = nn.RELU() 
        self.act = nn.ReLU() 
        
        self.fc1 = nn.Linear(1536, common_dim).double()
        self.fc2 = nn.Linear(128, common_dim).double()
        self.fc3 = nn.Linear(128, common_dim).double()
        self.fc5 = nn.Linear(2304, common_dim).double()
        self.fc4 = nn.Linear(common_dim, common_dim).double()
        
        self.classify = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(common_dim,common_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.1)
        ).double()
         
    def forward(self, bones, emg,spetrogram):
        x1 = self.gcn(bones)  # out
        x2 = self.vit(emg.double())  # out
        x3 = self.spectrogram_model(spetrogram.double())
       
        emg1 = rearrange(emg, "b (a d) c ->b a (d c) ", a=5).double()
        bone1 = rearrange(bones, "b t n c -> b t (n c)").double()
        spetrogram1 = rearrange(rearrange(spetrogram, "b c a d ->b d (c a)"),"b (a e) f-> b a (e f)",a = 5).double()
       
        
        c1 = self.att1(bone1,emg1)
        c2 = self.att2(bone1,spetrogram1)
        c3 = self.att3(spetrogram1,emg1) 
       
        x5 = self.vitForEMGandBone(torch.concat(
            [c1,c2,c3], dim=-1).double())  # out
        
        
        x5 = torch.concat([x1, x2,x3,x5], dim=-1)
       
        # common dim
        x1 = self.fc1(x1.double())
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x5 = self.fc5(x5)
        
        x1 = self.fc4(x1)
        x2 = self.fc4(x2)
        x3 = self.fc4(x3)
        x5 = self.fc4(x5)
        
        # classify
        a1 = self.classify(x1) + x1
        a2 = self.classify(x2) + x2
        a3 = self.classify(x3) + x3
        a5 = self.classify(x5) + x5

        a1 = self.fc(a1)
        a2 = self.fc(a2)
        a3 = self.fc(a3)
        a5 = self.fc(a5)

        return [x1, x2,x3,x5], [a1, a2,a3,a5]  # video,emg,multimodal

def super_gmc_loss(criterion,prediction, target, batch_representations, temperature, batch_size, cl_rate=2, params=[1,0.8,0.8]):
    joint_mod_loss_sum = 0
    losses = []
    
    for mod in range(len(batch_representations) - 1):
        # Negative pairs: everything that is not in the current joint-modality pair
        out_joint_mod = torch.cat(
            [batch_representations[-1], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
            mask_joint_mod
        ).view(2 * batch_size, -1)

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = torch.exp(
            torch.sum(
                batch_representations[-1] * batch_representations[mod], dim=-1
            )
            / temperature
        )
        # [2*B]
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
        loss_joint_mod = -torch.log(
            pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
        )
        losses.append(loss_joint_mod * params[mod])
        
        # print(torch.mean(loss_joint_mod).item())
    joint_mod_loss_sum = sum(losses) 
    
    
    supervised_loss = criterion(prediction[0], target) + criterion(prediction[1], target) + criterion(prediction[2], target) + criterion(prediction[3], target)
    joint_mod_loss_sum *= cl_rate
    
    L_GCM = torch.mean(joint_mod_loss_sum).item()
    L_classify = torch.mean(supervised_loss).item()
    L_GCM_vid = torch.mean(losses[0]).item()
    L_GCM_emg = torch.mean(losses[1]).item()
    L_GCM_spec = torch.mean(losses[2]).item()
    # print(L_GCM)
    # print(L_classify)
    

    loss = torch.mean(joint_mod_loss_sum + supervised_loss)
    # loss = torch.mean(supervised_loss)
    
    return loss,L_GCM,L_classify, L_GCM_vid, L_GCM_emg,L_GCM_spec

def train(train_loader, model, criterion, optimizer, device, T, loss_log):
    running_loss = 0
    loss_gcm = 0
    loss_classify = 0
    loss_gcm_vid = 0
    loss_gcm_emg = 0
    loss_gcm_spec = 0
    model.train()

    for videos, labels, emgs,sepctrograms in tqdm(train_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        sepctrograms = sepctrograms.to(device)
        emgs = emgs.to(device).double()
       
        # forward
        outputs, outputs2 = model(videos, emgs,sepctrograms)
            
        # backward
        loss,L_GMC,classification_loss, L_GMC_vid, L_GMC_emg,L_GCM_spec = super_gmc_loss(criterion,outputs2,labels,outputs,run.config['T'],len(labels), run.config['cl_rate'], params=[run.config['param_vid'], run.config['param_emg'],run.config['param_spec']])
    
        running_loss += loss.item()
        loss_gcm += L_GMC
        loss_gcm_vid += L_GMC_vid
        loss_gcm_emg += L_GMC_emg
        loss_gcm_spec += L_GCM_spec
        loss_classify += classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / (len(train_loader))
    loss_gcm = loss_gcm / (len(train_loader))
    loss_gcm_vid = loss_gcm_vid / (len(train_loader))
    loss_gcm_emg = loss_gcm_emg / (len(train_loader))
    loss_gcm_spec = loss_gcm_spec / (len(train_loader))
    loss_classify = loss_classify / (len(train_loader))
    loss_log[0].append(loss_gcm)
    loss_log[1].append(loss_classify)
    loss_log[2].append(loss_gcm_vid)
    loss_log[3].append(loss_gcm_emg)
    loss_log[4].append(loss_gcm_spec)
    loss_log[5].append(epoch_loss)

    return model, epoch_loss, optimizer, loss_log

def validate(valid_loader, model, criterion, device, T, val_loss_log):
    model.eval()
    running_loss = 0
    loss_gcm = 0
    loss_classify = 0
    loss_gcm_vid = 0
    loss_gcm_emg = 0
    loss_gcm_spec = 0

    for videos, labels, emgs,sepctrograms in tqdm(train_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        sepctrograms = sepctrograms.to(device)
        emgs = emgs.to(device).double()
       
        # forward
        outputs, outputs2 = model(videos, emgs,sepctrograms)

        # backward
        loss,L_GMC,classification_loss, L_GMC_vid, L_GMC_emg,L_GCM_spec = super_gmc_loss(criterion,outputs2,labels,outputs,run.config['T'],len(labels), run.config['cl_rate'], params=[run.config['param_vid'], run.config['param_emg'],run.config['param_spec']])
    
    
        running_loss += loss.item()
        loss_gcm += L_GMC
        loss_gcm_vid += L_GMC_vid
        loss_gcm_emg += L_GMC_emg
        loss_gcm_spec += L_GCM_spec
        loss_classify += classification_loss

    epoch_loss = running_loss / (len(train_loader))
    loss_gcm = loss_gcm / (len(train_loader))
    loss_gcm_vid = loss_gcm_vid / (len(train_loader))
    loss_gcm_emg = loss_gcm_emg / (len(train_loader))
    loss_gcm_spec = loss_gcm_spec / (len(train_loader))
    loss_classify = loss_classify / (len(train_loader))
    val_loss_log[0].append(loss_gcm)
    val_loss_log[1].append(loss_classify)
    val_loss_log[2].append(loss_gcm_vid)
    val_loss_log[3].append(loss_gcm_emg)
    val_loss_log[4].append(loss_gcm_spec)
    val_loss_log[5].append(epoch_loss)
    return model, epoch_loss, val_loss_log


def get_accuracy(model, data_loader, device, modality='multimodal'):
    correct = 0
    total = 0
    predicted_labels = []
    truth_labels = []

    model.eval()
    for videos, labels, emgs,sepctrograms in data_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        sepctrograms = sepctrograms.to(device)
        emgs = emgs.to(device).double()

        # forward
        _, outputs = model(videos, emgs,sepctrograms)
        if modality == 'multimodal':
            predicted = torch.argmax(torch.softmax(outputs[-1], 1), 1)
        elif modality == 'emg':
            predicted = torch.argmax(torch.softmax(outputs[1], 1), 1)
        elif modality == "spectrogram":
            predicted = torch.argmax(torch.softmax(outputs[2], 1), 1)
        else:
            predicted = torch.argmax(torch.softmax(outputs[0], 1), 1)

        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted)
        truth_labels.extend(labels)

    f1_macro = f1_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='macro')
    precision_score_f1 = precision_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='macro')
    recall_score_f1 = recall_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='macro')

    return correct/total, f1_macro, precision_score_f1, recall_score_f1


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(run.config["random_seed"])

trainset = ThreeModalData("data/new_data/new_train_files.pkl")
testset = ThreeModalData("data/new_data/new_test_files.pkl")
valset = ThreeModalData("data/new_data/new_val_files.pkl")

train_loader = DataLoader(trainset, batch_size=run.config['batch_size'],
                          drop_last=False, num_workers=5, shuffle=True)
valid_loader = DataLoader(valset, batch_size=run.config['batch_size'],
                          drop_last=False, num_workers=5 )
test_loader = DataLoader(testset, batch_size=run.config['batch_size'],
                         drop_last=False, num_workers=5)


device = run.config['device']
model = MultiModal(
    common_dim=run.config['common_dim'], n_classes=run.config['n_classes']).to(device)
# model.load_state_dict(torch.load("log/Missing_modal/best_model8.pth"))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters())


epochs = run.config['epochs']
train_losses = []
valid_losses = []
train_accuracy = []
val_accuracy = []

train_score_log = [[], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]]
val_score_log = [[], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]]
test_score_log =[[], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]]


loss_log = [[], [], [], [], [],[]]
val_loss_log =  [[], [], [], [], [],[]]


best_f1 = -1000

T = run.config['T']


log = Log("log/ThreeModalities", "multimodal")


for epoch in range(epochs):
    # training
    with torch.autograd.detect_anomaly():
        model, train_loss, optimizer, loss_log = train(
            train_loader, model, criterion, optimizer, device, T, loss_log)
    print("train_loss", train_loss)

    # validation
    with torch.no_grad():
        model, valid_loss, val_loss_log = validate(
            valid_loader, model, criterion, device, T, val_loss_log)
    print(valid_loss)

    # train_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
    #     model, train_loader, device, modality="vid")
    # train_score_log[0].append(train_acc)
    # train_score_log[1].append(f1_score_macro)
    # train_score_log[2].append(precision_score_macro)
    # train_score_log[3].append(recall_score_macro)

    # train_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
    #     model, train_loader, device, modality="emg")
    # train_score_log[4].append(train_acc)
    # train_score_log[5].append(f1_score_macro)
    # train_score_log[6].append(precision_score_macro)
    # train_score_log[7].append(recall_score_macro)
    # print("train acc", train_acc)

    # train_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
    #     model, train_loader, device, modality="multimodal")
    # train_score_log[8].append(train_acc)
    # train_score_log[9].append(f1_score_macro)
    # train_score_log[10].append(precision_score_macro)
    # train_score_log[11].append(recall_score_macro)

    # print("train acc", train_acc)

    val_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, valid_loader, device, modality="vid")
    val_score_log[0].append(val_acc)
    val_score_log[1].append(f1_score_macro)
    val_score_log[2].append(precision_score_macro)
    val_score_log[3].append(recall_score_macro)

    val_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, valid_loader, device, modality="emg")
    val_score_log[4].append(val_acc)
    val_score_log[5].append(f1_score_macro)
    val_score_log[6].append(precision_score_macro)
    val_score_log[7].append(recall_score_macro)
    print("Val acc", val_acc)
    
    val_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, valid_loader, device, modality="spectrogram")
    val_score_log[8].append(val_acc)
    val_score_log[9].append(f1_score_macro)
    val_score_log[10].append(precision_score_macro)
    val_score_log[11].append(recall_score_macro)
    print("Val acc", val_acc)

    val_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, valid_loader, device, modality="multimodal")
    val_score_log[12].append(val_acc)
    val_score_log[13].append(f1_score_macro)
    val_score_log[14].append(precision_score_macro)
    val_score_log[15].append(recall_score_macro)

    print("Val acc", val_acc)

    # save f1 score
    if best_f1 < f1_score_macro:
        torch.save(model.state_dict(),
                   f"log/ThreeModalities/best_model{epoch}.pth")
        best_f1 = f1_score_macro

     # get the bone accuracy
    test_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, test_loader, device, modality="vid")
    test_score_log[0].append(test_acc)
    test_score_log[1].append(f1_score_macro)
    test_score_log[2].append(precision_score_macro)
    test_score_log[3].append(recall_score_macro)
    print("test acc", test_acc)
    

    # get the emg accuracy
    test_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, test_loader, device, modality="emg")
    test_score_log[4].append(test_acc)
    test_score_log[5].append(f1_score_macro)
    test_score_log[6].append(precision_score_macro)
    test_score_log[7].append(recall_score_macro)
    print("test acc", test_acc)
    
    # get the emg accuracy
    test_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, test_loader, device, modality="spectrogram")
    test_score_log[8].append(test_acc)
    test_score_log[9].append(f1_score_macro)
    test_score_log[10].append(precision_score_macro)
    test_score_log[11].append(recall_score_macro)
    print("test acc", test_acc)
    

    # get the multimodal accuracy
    test_acc, f1_score_macro, precision_score_macro, recall_score_macro = get_accuracy(
        model, test_loader, device, modality="multimodal")
    test_score_log[12].append(test_acc)
    test_score_log[13].append(f1_score_macro)
    test_score_log[14].append(precision_score_macro)
    test_score_log[15].append(recall_score_macro)
    print("test acc", test_acc)
    # wandb.login(
    #     key='9bce1a84793dd8652665e9c5a731d2f7775245ad',
    #     relogin=True
    # )
    wandb.log({
        "Train loss": wandb.plot.line_series(
            xs=range(len(loss_log[0])),
            ys=loss_log,
            keys=["L_GCM", "L_classify","loss_gcm_vid","loss_gcm_emg","loss_gcm_spectrogram", "Loss"],
            title="Train loss",
            xname="x epochs"
        ),
        "Val loss": wandb.plot.line_series(
            xs=range(len(val_loss_log[0])),
            ys=val_loss_log,
            keys=["L_GCM", "L_classify","loss_gcm_vid","loss_gcm_emg","loss_gcm_spectrogram", "Loss"],
            title="Val loss",
            xname="x epochs"
        ),
        # "Train Accuracy Video": wandb.plot.line_series(
        #     xs=range(len(train_score_log[0])),
        #     ys=[train_score_log[0], train_score_log[1], train_score_log[2],train_score_log[3]],
        #     keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
        #     title="Train Accuracy Video",
        #     xname="x epochs"),
        # "Train Accuracy EMG": wandb.plot.line_series(
        #     xs=range(len(train_score_log[0])),
        #     ys=[train_score_log[4], train_score_log[5], train_score_log[6],train_score_log[7]],
        #     keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
        #     title="Train Accuracy EMG",
        #     xname="x epochs"),
        # "Train Accuracy Multimodal": wandb.plot.line_series(
        #     xs=range(len(train_score_log[0])),
        #     ys=[train_score_log[8], train_score_log[9], train_score_log[10],train_score_log[11]],
        #     keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
        #     title="Train Accuracy Multimodal",
        #     xname="x epochs"),

        "Val Accuracy Video": wandb.plot.line_series(
            xs=range(len(val_score_log[0])),
            ys=[val_score_log[0], val_score_log[1], val_score_log[2],val_score_log[3]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Val Accuracy Video",
            xname="x epochs"),
        "Val Accuracy EMG": wandb.plot.line_series(
            xs=range(len(val_score_log[0])),
            ys=[val_score_log[4], val_score_log[5], val_score_log[6],val_score_log[7]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Val Accuracy EMG",
            xname="x epochs"),
        "Val Accuracy Spectrogram": wandb.plot.line_series(
            xs=range(len(val_loss_log[0])),
            ys=[val_score_log[8], val_score_log[9], val_score_log[10],val_score_log[11]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Val Accuracy Spectrogram",
            xname="x epochs"),
         "Val Accuracy Multimodal": wandb.plot.line_series(
            xs=range(len(val_loss_log[0])),
            ys=[val_score_log[12], val_score_log[13], val_score_log[14],val_score_log[15]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Val Accuracy Multimodal",
            xname="x epochs"),
        
        
        "Test Accuracy Video": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[0], test_score_log[1], test_score_log[2],test_score_log[3]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Test Accuracy Video",
            xname="x epochs"),
        "Test Accuracy EMG": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[4], test_score_log[5], test_score_log[6],test_score_log[7]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Test Accuracy EMG",
            xname="x epochs"),
        "Test Accuracy Spectrogram": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[8], test_score_log[9], test_score_log[10],test_score_log[11]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Test Accuracy Spectrogram",
            xname="x epochs"),
         "Test Accuracy Multimodal": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[12], test_score_log[13], test_score_log[14],test_score_log[15]],
            keys=["Accuracy", "f1_score_macro", "precision_score_macro","recall_score_macro"],
            title="Test Accuracy EMG",
            xname="x epochs"),



    })

    # log.save_training_log(train_losses, train_accuracy,
    #                       train_f1score_weighted, train_f1score_macro)
    # log.save_val_log(valid_losses, val_accuracy,
    #                  val_f1scroe_weighted, val_f1scroe_macro)
    # log.save_test_log(test_log)

wandb.run.finish()
