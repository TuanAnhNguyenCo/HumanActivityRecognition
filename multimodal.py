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


class MultiModal(nn.Module):
    def __init__(self):
        super(MultiModal, self).__init__()
        self.gcn = GGCN(find_adjacency_matrix(), 41,
                        [3, 9], [9, 16, 32, 64], 0.0)
        self.vit = ViT(emg_size=(44100*0.2, 8), patch_height=60, num_classes=41, dim=126,
                       depth=3, mlp_dim=512, heads=8, pool='cls', dropout=0.3, emb_dropout=0.5).double()
        self.crossAtt1 = CrossAttention(63, 14112, 512).double()
        self.crossAtt2 = CrossAttention(14112, 63, 512).double()
        self.vitForEMGandBone = ViTForEMGAndBone(
            1024,  41, 512, 3, 8, 512, pool='cls', dim_head=64, dropout=0., emb_dropout=0.).double()

    def forward(self, bones, emg):

        x1 = self.gcn(bones)
        x2 = self.vit(emg.double())
        emg1 = rearrange(emg, "b (a d) c ->b a (d c) ", a=5).double()
        bone1 = rearrange(bones, "b t n c -> b t (n c)").double()
        x3 = self.crossAtt1(bone1, emg1)
        x4 = self.crossAtt2(emg1, bone1)
        x5 = self.vitForEMGandBone(torch.concat([x3, x4], dim=-1).double())
        return x2+x5 + x1


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

    f1_weighted = f1_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='weighted')
    f1_micro = f1_score(torch.tensor(truth_labels).cpu().data.numpy(), torch.tensor(
        predicted_labels).cpu().data.numpy(), average='weighted')
    return correct*100/total, f1_weighted, f1_micro


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(20)

trainset = MultiModalData("data/new_data/train_files.pkl")
testset = MultiModalData("data/new_data/test_files.pkl")
valset = MultiModalData("data/new_data/val_files.pkl")

train_loader = DataLoader(trainset, batch_size=256,
                          drop_last=False, num_workers=10)
valid_loader = DataLoader(valset, batch_size=256,
                          drop_last=False, num_workers=10)
test_loader = DataLoader(testset, batch_size=256,
                         drop_last=False, num_workers=10)


device = 'cuda:0'
model = MultiModal().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


epochs = 100
train_losses = []
valid_losses = []
train_accuracy = []
val_accuracy = []

train_f1score_weighted = []
val_f1scroe_weighted = []

train_f1score_micro = []
val_f1scroe_micro = []

test_log = []

best_f1 = -1000

log = Log("log/Multimodal2", "multimodal")


for epoch in range(epochs):
    # training
    model, train_loss, optimizer = train(
        train_loader, model, criterion, optimizer, device)

    # validation
    with torch.no_grad():
        model, valid_loss = validate(valid_loader, model, criterion, device)
    train_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, train_loader, device)
    # save f1 score
    train_f1score_weighted.append(f1_score_weighted)
    train_f1score_micro.append(f1_score_micro)

    val_acc, f1_score_weighted, f1_score_micro = get_accuracy(
        model, valid_loader, device)
    # save f1 score
    if best_f1 < f1_score_micro:
        torch.save(model.state_dict(),
                   f"log/Multimodal2/best_model{epoch}.pth")

    val_f1scroe_weighted.append(f1_score_weighted)
    val_f1scroe_micro.append(f1_score_micro)
    print("Epoch {} --- Train loss = {} --- Valid loss = {} -- Train set accuracy = {} % Valid set Accuracy = {} %".format
          (epoch+1, train_loss, valid_loss, train_acc, val_acc))
    # save loss value
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # save accuracy
    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)

    test_log.append(get_accuracy(model, test_loader, device))

    log.save_training_log(train_losses, train_accuracy,
                          train_f1score_weighted, train_f1score_micro)
    log.save_val_log(valid_losses, val_accuracy,
                     val_f1scroe_weighted, val_f1scroe_micro)
    log.save_test_log(test_log)
