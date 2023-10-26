import torch
import torch.nn.functional as F
from typing import Optional
import random
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import cv2
import numpy as np
import math
import mediapipe as mp
from matplotlib import pyplot as plt
import glob
from util.img2bone import HandDetector
import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from tqdm.auto import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from loader.dataloader import SkeletonAndEMGData1
from sklearn.metrics import precision_score, recall_score, f1_score
from util.log import Log
from trainer import train, validate, get_accuracy


# helpers
device = "cuda:1"


class SE(nn.Module):
    def __init__(self, out_dim, hidden_dim, expansion=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, int(hidden_dim * expansion), bias=False),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * expansion), out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant, irrelevant)
    apr, api = module.ap.attrib(relevant, irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)


def attrib_adaptiveconcatpool(self, relevant, irrelevant):
    return cd_adaptiveconcatpool(relevant, irrelevant, self)


class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

    def attrib(self, relevant, irrelevant):
        return attrib_adaptiveconcatpool(self, relevant, irrelevant)


def maxpool(kernel, stride):
    return nn.MaxPool1d(kernel, stride=stride, padding=(kernel-1)//2)


def conv(in_planes, out_planes, kernel_size=3, stride=1, groups=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False, groups=groups)


def avgpool(kernel, stride):
    return nn.AvgPool1d(kernel, stride=stride, padding=(kernel-1)//2)


t = 4   # number of branches


class Model_conv_block(nn.Module):
    def __init__(self, ic, oc_filter, bottle_neck=32, stride=1):
        super().__init__()
        self.s_conv = nn.Conv1d(ic, bottle_neck, 1, stride=1)
        self.se = SE(bottle_neck, bottle_neck)

        self.s_convb = conv(bottle_neck, oc_filter, 39, stride=stride)
        self.s_convm = conv(bottle_neck, oc_filter, 19, stride=stride)
        self.s_convs = conv(bottle_neck, oc_filter, 9, stride=stride)

        self.t_pool = maxpool(3, stride=stride)
        self.t_conv = conv(ic, oc_filter, 3)

        self.skip = conv(ic, oc_filter*t, 3, stride=stride)

        self.bn_relu = nn.Sequential(nn.BatchNorm1d(oc_filter*t), nn.ReLU())

    def forward(self, x):
        bs = []
        # spatial
        b = self.s_conv(x)
        b = self.se(b)
        bs.append(self.s_convb(b))
        bs.append(self.s_convm(b))
        bs.append(self.s_convs(b))
        bs.append(self.t_conv(self.t_pool(x)))

        bs = torch.concatenate(bs, dim=1)
        return self.bn_relu(bs + self.skip(x))


class IncepSE(nn.Module):
    def __init__(self, num_class=5, ic=12, oc_filter=32, bottle_neck=32, depth=7, stride=1, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[Model_conv_block(ic=oc_filter*t*2 if i >= depth-1 else oc_filter*t,
                                                       oc_filter=2*oc_filter if i >= depth-2 else oc_filter,
                                                       bottle_neck=2*bottle_neck if i >= depth-2 else bottle_neck,
                                                       stride=2 if i >= depth-2 else stride) for i in range(depth-1)])
        self.blocks.insert(0, Model_conv_block(
            ic=ic, oc_filter=oc_filter, bottle_neck=bottle_neck, stride=stride))

        self.pool = AdaptiveConcatPool1d(1)
        self.downsample = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=45, stride=33, groups=8),
            nn.BatchNorm1d(8),
            nn.GELU()
        )
        # self.downsample = nn.AdaptiveAvgPool1d(1024)
        classify = [nn.Flatten()]
        classify.append(nn.Dropout(dropout))
        classify.append(nn.Linear(4*oc_filter*t, num_class))

        self.classify = nn.Sequential(*classify)

    def forward(self, x):

        x = self.downsample(x)

        x = self.blocks(x)
        x = self.pool(x)
       
        return self.classify(x)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(20)

data = []
data.extend(glob.glob('data/new_data/emg_data/*'))


index = np.random.permutation(len(data))
data = np.array(data)[index]
train_size = 12000
test_size = 4500
val_size = data.shape[0]-train_size - test_size
trainset = data[:train_size]
testset = data[train_size:train_size+test_size]
valset = data[train_size+test_size:]

train_set = SkeletonAndEMGData1(trainset)
test_set = SkeletonAndEMGData1(testset)
val_set = SkeletonAndEMGData1(valset)


train_loader = DataLoader(train_set, batch_size=32,
                          drop_last=True, num_workers=3, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=32,
                          drop_last=True, num_workers=3)
test_loader = DataLoader(test_set, batch_size=32,
                         drop_last=True, num_workers=3)

# model = model = IncepSE(num_class=41,ic = 8,depth = 5,bottle_neck=32,oc_filter=32,dropout=0).to(device).double()
model = IncepSE(num_class=41, ic=8, depth=3, bottle_neck=30,
                oc_filter=30, dropout=0.3).to(device).double()

log = Log("log/InceptionNet", "vit_emg")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
stepLR = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.8)
print(3,32,32,0,"step",20,"gamma",0.8)
epochs = 500
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

for epoch in range(epochs):
    # training
    model, train_loss, optimizer = train(
        train_loader, model, criterion, optimizer, device)
    stepLR.step()

    # validation
    with torch.no_grad():
        model, valid_loss = validate(valid_loader, model, criterion, device)
    stepLR.step(valid_loss)
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
                   f"log/InceptionNet/best_model{epoch}.pth")
        log.save_model(model)
        best_f1 = f1_score_micro
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
