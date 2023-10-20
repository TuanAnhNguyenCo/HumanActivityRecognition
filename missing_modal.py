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
        "epochs": 10,
        'random_seed': 20,
        "device": "cuda:1",
        "common_dim": 128,
        "n_classes": 41,
        "batch_size": 128,
        "T": 0.1
    })


class MultiModal(nn.Module):
    def __init__(self, common_dim, n_classes):
        super(MultiModal, self).__init__()
        self.gcn = GGCN(find_adjacency_matrix(), 41,
                        [3, 9], [9, 16, 32, 64], run.config["device"], 0.0)
        self.vit = ViT(emg_size=(44100*0.2, 8), patch_height=60, num_classes=41, dim=128,
                       depth=3, mlp_dim=512, heads=8, pool='cls', dropout=0., emb_dropout=0.3).double()
        self.crossAtt1 = CrossAttention(63, 14112, 512).double()
        self.crossAtt2 = CrossAttention(14112, 63, 512).double()
        self.vitForEMGandBone = ViTForEMGAndBone(
            1024,  41, 512, 3, 8, 512, pool='cls', dim_head=64, dropout=0., emb_dropout=0.).double()

        self.fc = nn.Linear(common_dim, n_classes).double()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(1536, common_dim).double()
        self.fc2 = nn.Linear(128, common_dim).double()
        self.fc3 = nn.Linear(512, common_dim).double()
        self.fc4 = nn.Linear(common_dim, common_dim).double()
        self.fc5 = nn.Linear(common_dim*3, common_dim).double()
        self.drop = nn.Dropout(p=0.3)

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

        x1 = self.fc1(x1.double())
        x2 = self.fc2(x2)
        x5 = self.fc3(x5)
        # x5 = x1+x2+x5
        x5 = torch.concat([x1, x2, x5], dim=-1)
        x5 = self.fc5(self.act(x5))

        # x1 = self.act(x1)
        # x2 = self.act(x2)
        # x5 = self.act(x5)

        # x1 = self.fc4(x1)
        # x2 = self.fc4(x2)
        # x5 = self.fc4(x5)

        a1 = self.act(x1)
        a2 = self.act(x2)
        a5 = self.act(x5)

        a1 = self.fc(a1)
        a2 = self.fc(a2)
        a5 = self.fc(a5)

        return [x1, x2, x5], [a1, a2, a5]  # video,emg,multimodal


def train(train_loader, model, criterion, optimizer, device, T, loss_log):
    running_loss = 0
    loss_gcm = 0
    loss_classify = 0
    model.train()

    for videos, labels, emgs in tqdm(train_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()
        L_GMC = 0

        # forward
        outputs, outputs2 = model(videos, emgs)
        multi = outputs[-1].clone().detach()
        for i in range(2):

            x1 = torch.exp(torch.einsum(
                "ab,cb -> ac", outputs[i], multi)/T)
            x2 = torch.exp(torch.einsum(
                "ab,cb -> ac", outputs[i], outputs[i])/T)
            x3 = torch.exp(torch.einsum(
                "ab,cb -> ac", outputs[-1], outputs[-1])/T)
            mask_joint_mod = (
                torch.ones_like(x1)
                - torch.eye(x1.shape[0], device=x1.device)
            )

            o1 = x1*mask_joint_mod
            o2 = x2*mask_joint_mod
            o3 = x3*mask_joint_mod
            o4 = x1*torch.eye(x1.shape[0], device=x1.device)
            o4 = torch.sum(o4, dim=-1)

            o = torch.concat([o1, o2, o3], dim=-1)
            o = torch.sum(o, dim=-1)
            
            o = -torch.log(o4/o)

            o = torch.mean(o, dim=0)
            
            

            L_GMC += o
          

        # L_GMC /= (2*outputs[0].shape[0])
        L_GMC /=2
        classification_loss = 2*criterion(outputs2[-1], labels)
        loss = classification_loss + L_GMC
        running_loss += loss.item()
        loss_gcm += L_GMC.item()
        loss_classify += classification_loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / (len(train_loader))
    loss_gcm = loss_gcm / (len(train_loader))
    loss_classify = loss_classify / (len(train_loader))
    loss_log[0].append(loss_gcm)
    loss_log[1].append(loss_classify)
    loss_log[2].append(epoch_loss)

    return model, epoch_loss, optimizer, loss_log


def validate(valid_loader, model, criterion, device, T, val_loss_log):
    model.eval()
    running_loss = 0
    loss_gcm = 0
    loss_classify = 0

    for videos, labels, emgs in tqdm(valid_loader):

        videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()
        L_GMC = 0
        # forward
        outputs, outputs2 = model(videos, emgs)
        for i in range(2):

            x1 = torch.exp(torch.einsum(
                "ab,cb -> ac", outputs[i], outputs[-1])/T)
            x2 = torch.exp(torch.einsum(
                "ab,cb -> ac", outputs[i], outputs[i])/T)
            x3 = torch.exp(torch.einsum(
                "ab,cb -> ac", outputs[-1], outputs[-1])/T)
            mask_joint_mod = (
                torch.ones_like(x1)
                - torch.eye(x1.shape[0], device=x1.device)
            )

            o1 = x1*mask_joint_mod
            o2 = x2*mask_joint_mod
            o3 = x3*mask_joint_mod
            o4 = x1*torch.eye(x1.shape[0], device=x1.device)
            o4 = torch.sum(o4, dim=-1)

            o = torch.concat([o1, o2, o3], dim=-1)
            o = torch.sum(o, dim=-1)

            o = -torch.log(o4/o)

            o = torch.sum(o, dim=0)

            L_GMC += o

        L_GMC /= (2*outputs[0].shape[0])
        classification_loss = criterion(outputs2[-1], labels)
        loss = classification_loss + L_GMC
        running_loss += loss.item()
        loss_gcm += L_GMC.item()
        loss_classify += classification_loss.item()

    epoch_loss = running_loss / (len(train_loader))
    loss_gcm = loss_gcm / (len(train_loader))
    loss_classify = loss_classify / (len(train_loader))
    val_loss_log[0].append(loss_gcm)
    val_loss_log[1].append(loss_classify)
    val_loss_log[2].append(epoch_loss)
    return model, epoch_loss, val_loss_log


def get_accuracy(model, data_loader, device, modality='multimodal'):
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
        _, outputs = model(videos, emgs)
        if modality == 'multimodal':
            predicted = torch.argmax(torch.softmax(outputs[-1], 1), 1)
        elif modality == 'emg':
            predicted = torch.argmax(torch.softmax(outputs[1], 1), 1)
        else:
            predicted = torch.argmax(torch.softmax(outputs[0], 1), 1)

        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted)
        truth_labels.extend(labels)

    f1_weighted = f1_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='weighted')
    f1_micro = f1_score(torch.tensor(truth_labels).cpu().data.numpy(), torch.tensor(
        predicted_labels).cpu().data.numpy(), average='weighted')
    return correct/total, f1_weighted, f1_micro


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(run.config["random_seed"])

trainset = MultiModalData("data/new_data/train_files.pkl")
testset = MultiModalData("data/new_data/test_files.pkl")
valset = MultiModalData("data/new_data/val_files.pkl")

train_loader = DataLoader(trainset, batch_size=run.config['batch_size'],
                          drop_last=False, num_workers=10, shuffle=True)
valid_loader = DataLoader(valset, batch_size=run.config['batch_size'],
                          drop_last=False, num_workers=10 )
test_loader = DataLoader(testset, batch_size=run.config['batch_size'],
                         drop_last=False, num_workers=10)


device = run.config['device']
model = MultiModal(
    common_dim=run.config['common_dim'], n_classes=run.config['n_classes']).to(device)
model.load_state_dict(torch.load("log/Missing_modal/best_model9.pth"))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters())


epochs = run.config['epochs']
train_losses = []
valid_losses = []
train_accuracy = []
val_accuracy = []


# train_log_video = [[],[],[]]
# train_log_emg = [[],[],[]]
# train_log_multimodal = [[],[],[]]


# val_log_video = [[],[],[]]
# val_log_emg = [[],[],[]]
# val_log_multimodal = [[],[],[]]

train_score_log = [[], [], [], [], [], [], [], [], []]
val_score_log = [[], [], [], [], [], [], [], [], []]
test_score_log = [[], [], [], [], [], [], [], [], []]


loss_log = [[], [], []]
val_loss_log = [[], [], []]

# test_log_video = [[],[],[]]
# test_log_emg = [[],[],[]]
# test_log_multimodal = [[],[],[]]

best_f1 = -1000

T = run.config['T']


log = Log("log/Missing_modal", "multimodal")


for epoch in range(epochs):
    # training
    model, train_loss, optimizer, loss_log = train(
        train_loader, model, criterion, optimizer, device, T, loss_log)
    print("train_loss", train_loss)

    # validation
    with torch.no_grad():
        model, valid_loss, val_loss_log = validate(
            valid_loader, model, criterion, device, T, val_loss_log)

    train_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, train_loader, device, modality="vid")
    train_score_log[0].append(train_acc)
    train_score_log[1].append(f1_score_weighted)
    train_score_log[2].append(f1_score_micro)

    train_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, train_loader, device, modality="emg")
    train_score_log[3].append(train_acc)
    train_score_log[4].append(f1_score_weighted)
    train_score_log[5].append(f1_score_micro)

    train_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, train_loader, device, modality="multimodal")
    train_score_log[6].append(train_acc)
    train_score_log[7].append(f1_score_weighted)
    train_score_log[8].append(f1_score_micro)

    print("train acc", train_acc)

    val_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, valid_loader, device, modality="vid")
    val_score_log[0].append(val_acc)
    val_score_log[1].append(f1_score_weighted)
    val_score_log[2].append(f1_score_micro)

    val_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, valid_loader, device, modality="emg")
    val_score_log[3].append(val_acc)
    val_score_log[4].append(f1_score_weighted)
    val_score_log[5].append(f1_score_micro)

    val_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, valid_loader, device, modality="multimodal")
    val_score_log[6].append(val_acc)
    val_score_log[7].append(f1_score_weighted)
    val_score_log[8].append(f1_score_micro)

    print("Val acc", val_acc)

    # save f1 score
    if best_f1 < f1_score_micro:
        torch.save(model.state_dict(),
                   f"log/Missing_modal/best_model{epoch}.pth")
        best_f1 = f1_score_micro

     # get the bone accuracy
    test_acc, tesst_f1_score_weighted, test_f1_score_micro = get_accuracy(
        model, test_loader, device, modality="vid")
    test_score_log[0].append(test_acc)
    test_score_log[1].append(tesst_f1_score_weighted)
    test_score_log[2].append(test_f1_score_micro)

    # get the emg accuracy
    test_acc, tesst_f1_score_weighted, test_f1_score_micro = get_accuracy(
        model, test_loader, device, modality="emg")
    test_score_log[3].append(test_acc)
    test_score_log[4].append(tesst_f1_score_weighted)
    test_score_log[5].append(test_f1_score_micro)

    # get the multimodal accuracy
    test_acc, tesst_f1_score_weighted, test_f1_score_micro = get_accuracy(
        model, test_loader, device, modality="multimodal")
    test_score_log[6].append(test_acc)
    test_score_log[7].append(tesst_f1_score_weighted)
    test_score_log[8].append(test_f1_score_micro)
    print("test acc", test_acc)
    wandb.log({
        "Train loss": wandb.plot.line_series(
            xs=range(len(loss_log[0])),
            ys=loss_log,
            keys=["L_GCM", "L_classify", "Loss"],
            title="Train loss",
            xname="x epochs"
        ),
        "Val loss": wandb.plot.line_series(
            xs=range(len(val_loss_log[0])),
            ys=val_loss_log,
            keys=["L_GCM", "L_classify", "Loss"],
            title="Val loss",
            xname="x epochs"
        ),
        "Train Accuracy Video": wandb.plot.line_series(
            xs=range(len(train_score_log[0])),
            ys=[train_score_log[0], train_score_log[1], train_score_log[2]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Train Accuracy Video",
            xname="x epochs"),
        "Train Accuracy EMG": wandb.plot.line_series(
            xs=range(len(train_score_log[0])),
            ys=[train_score_log[3], train_score_log[4], train_score_log[5]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Train Accuracy EMG",
            xname="x epochs"),
        "Train Accuracy Multimodal": wandb.plot.line_series(
            xs=range(len(train_score_log[0])),
            ys=[train_score_log[6], train_score_log[7], train_score_log[8]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Train Accuracy Multimodal",
            xname="x epochs"),

        "Val Accuracy Video": wandb.plot.line_series(
            xs=range(len(val_score_log[0])),
            ys=[val_score_log[0], val_score_log[1], val_score_log[2]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Val Accuracy Video",
            xname="x epochs"),
        "Val Accuracy EMG": wandb.plot.line_series(
            xs=range(len(val_score_log[0])),
            ys=[val_score_log[3], val_score_log[4], val_score_log[5]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Val Accuracy EMG",
            xname="x epochs"),
        "Val Accuracy Multimodal": wandb.plot.line_series(
            xs=range(len(val_loss_log[0])),
            ys=[val_score_log[6], val_score_log[7], val_score_log[8]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Val Accuracy Multimodal",
            xname="x epochs"),
        "Test Accuracy Video": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[0], test_score_log[1], test_score_log[2]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Test Accuracy Video",
            xname="x epochs"),
        "Test Accuracy EMG": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[3], test_score_log[4], test_score_log[5]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Test Accuracy EMG",
            xname="x epochs"),
        "Test Accuracy Multimodal": wandb.plot.line_series(
            xs=range(len(test_score_log[0])),
            ys=[test_score_log[6], test_score_log[7], test_score_log[8]],
            keys=["Accuracy", "F1_score_weighted", "F1_score_micro"],
            title="Test Accuracy Multimodal",
            xname="x epochs"),



    })

    # log.save_training_log(train_losses, train_accuracy,
    #                       train_f1score_weighted, train_f1score_micro)
    # log.save_val_log(valid_losses, val_accuracy,
    #                  val_f1scroe_weighted, val_f1scroe_micro)
    # log.save_test_log(test_log)

wandb.run.finish()
