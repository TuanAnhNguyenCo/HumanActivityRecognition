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
from loader.dataloader import SkeletonAndEMGData
from sklearn.metrics import precision_score, recall_score, f1_score
from util.log import Log
import wandb


# wandb.login(
#     key='0c9e631211d579a62fa94880e8e3efcf09cd66a0',
# )

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="Multimodal-HAR",
#     entity='aiotlab',
#     group='VIT_for_EMG',
#     name=f'VIT_for_EMG_lrate:{1e-3}',
#     # Track hyperparameters and run metadata
#     config={
#         "epochs": 30,
#         'random_seed': 20,
#         "n_classes": 41,
#         "batch_size": 128,
#         "device": 'cuda:1'
#     })


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, emg_size, patch_height, num_classes, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        emg_height, emg_width = pair(emg_size)

        num_patches = int(emg_height//patch_height)
        patch_dim = int(emg_width * patch_height)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) c -> b h (p1 c)', h=num_patches, c=emg_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # return self.mlp_head(x) # multimodal
        return x


def train(train_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()

    for videos, labels, emgs in tqdm(train_loader):

        # videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()

        # forward
        outputs = model(emgs)
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

        # videos = videos.to(device)
        labels = labels.to(device)
        emgs = emgs.to(device).double()

        # forward

        outputs = model(emgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    epoch_loss = running_loss / (len(valid_loader))
    return model, epoch_loss


def get_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    predicted_labels = []
    truth_labels = []

    with torch.no_grad():
        model.eval()
        for videos, labels, emgs in data_loader:
            # videos = videos.to(device)
            labels = labels.to(device)
            emgs = emgs.to(device).double()

            # forward
            outputs = model(emgs)
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


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True


# seed_everything(run.config['random_seed'])

# trainset = SkeletonAndEMGData("data/new_data/emg_train_files.pkl")
# testset = SkeletonAndEMGData("data/new_data/emg_test_files.pkl")
# valset = SkeletonAndEMGData("data/new_data/emg_val_files.pkl")

# train_loader = DataLoader(trainset, batch_size=run.config['batch_size'],
#                           drop_last=True, num_workers=3, shuffle=True)
# valid_loader = DataLoader(valset, batch_size=run.config['batch_size'],
#                           drop_last=True, num_workers=3 )
# test_loader = DataLoader(testset, batch_size=run.config['batch_size'],
#                          drop_last=True, num_workers=3)

# model = ViT(emg_size=(44100*0.2, 8), patch_height=60, num_classes=41,
#             dim=126, depth=3, mlp_dim=512, heads=8, pool='cls',
#             dropout=0.3, emb_dropout=0.5
#             ).to(run.config['device']).double()

# log = Log("log/VIT_EMG", "vit_emg")

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
# epochs = run.config['epochs']
# train_losses = []
# valid_losses = []
# train_accuracy = []
# val_accuracy = []

# train_score_log = [[], [], [], []]
# val_score_log = [[], [], [], [], []]
# test_score_log = [[], [], [], [], []]
# device = run.config['device']
# best_f1 = -1000

# log = Log("log/GCN", "gcn")


# for epoch in range(epochs):
#     # training

#     model, train_loss, optimizer = train(
#             train_loader, model, criterion, optimizer, device)
#     train_losses.append(train_loss)

#     # validation
#     with torch.no_grad():
#         model, valid_loss = validate(
#             valid_loader, model, criterion, device)
#     valid_losses.append(valid_loss)

#     # get train score
#     train_acc, f1_score_micro,precision_score_micro,recall_score_micro = get_accuracy(model, train_loader, device)
#     train_score_log[0].append(train_acc)
#     train_score_log[1].append(f1_score_micro)
#     train_score_log[2].append(precision_score_micro)
#     train_score_log[3].append(recall_score_micro)

#      # get val score
#     val_acc, f1_score_micro,precision_score_micro,recall_score_micro = get_accuracy(model, valid_loader, device)
#     val_score_log[0].append(val_acc)
#     val_score_log[1].append(f1_score_micro)
#     val_score_log[2].append(precision_score_micro)
#     val_score_log[3].append(recall_score_micro)

#     # save f1 score
#     if best_f1 < f1_score_micro:
#         torch.save(model.state_dict(),
#                    f"log/GCN/best_model{epoch}.pth")
#         best_f1 = f1_score_micro

#     # get test score
#     test_acc, f1_score_micro,precision_score_micro,recall_score_micro = get_accuracy(model, test_loader, device)
#     test_score_log[0].append(test_acc)
#     test_score_log[1].append(f1_score_micro)
#     test_score_log[2].append(precision_score_micro)
#     test_score_log[3].append(recall_score_micro)

#     wandb.log({
#         "Train loss": wandb.plot.line_series(
#             xs=range(len(train_losses)),
#             ys=[train_losses],
#             keys= ["Loss"],
#             title="Train loss",
#             xname="x epochs"
#         ),
#         "Val loss": wandb.plot.line_series(
#             xs=range(len(valid_losses)),
#             ys=[valid_losses],
#             keys=["Loss"],
#             title="Val loss",
#             xname="x epochs"
#         ),
#         "Train Score": wandb.plot.line_series(
#             xs=range(len(train_score_log[0])),
#             ys=[train_score_log[0], train_score_log[1], train_score_log[2],train_score_log[3]],
#             keys=["Accuracy", "F1_score_micro","precision_score_micro","recall_score_micro"],
#             title="Train Score",
#             xname="x epochs"),
#         "Val Score": wandb.plot.line_series(
#             xs=range(len(val_score_log[0])),
#             ys=[val_score_log[0], val_score_log[1], val_score_log[2],val_score_log[3]],
#             keys=["Accuracy", "F1_score_micro","precision_score_micro","recall_score_micro"],
#             title="Val Score",
#             xname="x epochs"),
#         "Test Score": wandb.plot.line_series(
#             xs=range(len(test_score_log[0])),
#             ys=[test_score_log[0], test_score_log[1], test_score_log[2],test_score_log[3]],
#             keys=["Accuracy", "F1_score_micro","precision_score_micro","recall_score_micro"],
#             title="Test Score",
#             xname="x epochs"),

#     })
# wandb.run.finish()
