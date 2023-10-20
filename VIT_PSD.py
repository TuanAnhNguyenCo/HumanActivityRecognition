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
from loader.dataloader import MultiModalData1
from sklearn.metrics import precision_score, recall_score, f1_score
from util.log import Log

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


# class ConvertPSDToIMG(nn.Module):
#     def __init__(self, device):
#         super().__init__()
#         self.convert_to_img = [nn.Sequential(
#             Rearrange("b a -> b 1 a "),
#             nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=1),
#             nn.InstanceNorm1d(64),
#             nn.ReLU(),
#             nn.Flatten(),
#             Rearrange("b (c d) -> b c d ", c=32),
#         ).double().to(device) for i in range(1)]

#     def forward(self, x):
#         b, c, _ = x.shape
#         new_img = torch.ones((b, c, 32, 32))
#         for i in range(c):
#             new_img[:, i] = self.convert_to_img[0](x[:, i])

#         return new_img.to(device).double()


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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., device):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        print(patch_height, patch_width)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width, c=channels),
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

        self.mlp_head = nn.Linear(dim, num_classes)

        # self.convert_to_img = ConvertPSDToIMG(device)

    def forward(self, x):

        # x = self.convert_to_img(pxx)
        # print()

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


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

    f1_weighted = f1_score(torch.tensor(truth_labels).cpu().data.numpy(
    ), torch.tensor(predicted_labels).cpu().data.numpy(), average='weighted')
    f1_micro = f1_score(torch.tensor(truth_labels).cpu().data.numpy(), torch.tensor(
        predicted_labels).cpu().data.numpy(), average='micro')
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


device = 'cuda:0'

data = []
data.extend(glob.glob('data/new_data/Scalogram/val2/*'))
data.extend(glob.glob('data/new_data/Scalogram/train2/*'))
data.extend(glob.glob('data/new_data/Scalogram/test2/*'))


index = np.random.permutation(len(data))
data = np.array(data)[index]
train_size = 12000
test_size = 3500
val_size = data.shape[0]-train_size - test_size
trainset = data[:train_size]
testset = data[train_size:train_size+test_size]
valset = data[train_size+test_size:]


train_set = MultiModalData1(trainset)
test_set = MultiModalData1(testset)
val_set = MultiModalData1(valset)


train_loader = DataLoader(train_set, batch_size=256,
                          drop_last=False, num_workers=0)
valid_loader = DataLoader(val_set, batch_size=256,
                          drop_last=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256,
                         drop_last=False, num_workers=0)

model = ViT(image_size=(224, 224), patch_size=(28, 28), num_classes=41, dim=128, depth=3,
            heads = 8, mlp_dim=256, pool='cls', channels=8, dim_head=64, dropout=0, emb_dropout=0, device=device
            ).to(device).double()

log = Log("log/VIT_Spec", "vit_emg")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
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
        # torch.save(model.state_dict(),"log/test/demo.pth")
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
