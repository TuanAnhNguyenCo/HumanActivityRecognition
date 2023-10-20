import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat
from trainer import train,validate,get_accuracy
from torch.utils.data import Dataset, DataLoader
from util.log import Log
from loader.dataloader import SkeletonAndEMGData
import glob
import numpy as np
import random
import os

class MultiKernelLayer(nn.Module):
    def __init__(self,width) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,1,1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Conv2d(1,1,(2,width)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,1,3,padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Conv2d(1,1,(2,width)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1,1,5,padding=2),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Conv2d(1,1,(2,width)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x = x.unsqueeze(dim = 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        return (x1 + x2 + x3).squeeze(dim= -1)
    
class ConvolutionComponent(nn.Module):
    def __init__(self,width,t) -> None:
        super().__init__()
        self.conv = nn.ModuleList([MultiKernelLayer(width) for i in range(t)])
    def forward(self,x):
       
        results = self.conv[0](x[:,0])
        for i,conv in enumerate(self.conv[1:]):
            results = torch.concat((results,conv(x[:,i+1])),dim = 1)
        return results
class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.positional_encoding[:batch_size, :seq_len, :]
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.att =  Attention(64, heads=8,
                                        dim_head=64, dropout=0)
        self.ff_layer = FeedForward(64, 64, dropout=0)
        self.midLayerNorm = nn.LayerNorm(64, eps=1e-05)
        self.outLayerNorm = nn.LayerNorm(64, eps=1e-05)
    def forward(self,x):
        hidden_states = self.att(x)
        hidden_states = self.midLayerNorm(x+hidden_states)
        # feed-forward layer
        ff_hidden_states = self.ff_layer(hidden_states)

        # add & norm layer
        output_hidden_states = self.outLayerNorm(hidden_states+ff_hidden_states)
        return output_hidden_states
        
    
class Demo(nn.Module):
    def __init__(self,device) -> None:
        super().__init__()
        self.conv = ConvolutionComponent(3,64)
        self.linear = nn.Linear(2,64)
        self.position_encoding = AbsolutePositionalEncoder(64,64)
        self.encoder = TransformerEncoderBlock()
        self.LSTM = nn.LSTM(64,64,1,True,True)
        self.downsample = nn.AdaptiveAvgPool1d(512)
        self.ll = nn.Linear(8,64)
        self.classify = nn.Linear(64,41)
        self.device = device
        self.ll1 = nn.Linear(8820,1024)
        self.act = nn.Sigmoid()
        self.act1 = nn.LeakyReLU()
        self.ll2 = nn.Linear(1024,2048)
        self.ll3 = nn.Linear(2048,1024)
        
    def forward(self,x):
        x1 = self.ll1(x)
        x2 = self.act(self.ll3(self.act1(self.ll2(x1))))
        x = x1*x2
        old = rearrange(x," a b c -> a c b")
        x = torch.concat((x,torch.zeros((x.shape[0],1,x.shape[-1])).to(self.device)),dim = 1)
        x = rearrange(x,"b (p k) l -> b l p k",p = 3)
        
        x = self.conv(x)
        x = self.position_encoding(x).to(self.device) + self.linear(x)
        x = self.encoder(x)
        x,_ = self.LSTM(x)
        
        ll = self.ll(old)
        x = torch.sum(x,dim = 1).squeeze() + torch.sum(ll,dim = 1) + torch.rand((ll.shape[0],ll.shape[-1])).to(self.device)
        return self.classify(x)
        
        
device = 'cuda:0'
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
data.extend(glob.glob('data/new_data/val2/*'))
data.extend(glob.glob('data/new_data/train2/*'))
data.extend(glob.glob('data/new_data/test2/*'))


index = np.random.permutation(len(data))
data = np.array(data)[index]
train_size = 12000
test_size = 3500
val_size = data.shape[0]-train_size - test_size
trainset = data[:train_size]
testset = data[train_size:train_size+test_size]
valset = data[train_size+test_size:]

train_set = SkeletonAndEMGData(trainset)
test_set = SkeletonAndEMGData(testset)
val_set = SkeletonAndEMGData(valset)


train_loader = DataLoader(train_set, batch_size=256,
                          drop_last=False, num_workers=10,shuffle=True)
valid_loader = DataLoader(val_set, batch_size=256,
                          drop_last=False, num_workers=10)
test_loader = DataLoader(test_set, batch_size=256,
                         drop_last=False, num_workers=10)

model = Demo(device).to(device).double()

log = Log("log/InceptionNet", "vit_emg")

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
