import torch
from torch import nn


class Block(nn.Module):
    def __init__(self,in_channels,out_channels,dropout = 0,kernel_size = 3,padding = 1):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size,padding = padding),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2,stride = 2),
        )
    def forward(self,x):
        return self.block(x)

class HandSignCNNModel(nn.Module):
    def __init__(self,n_class):
        super(HandSignCNNModel, self).__init__()
        self.block1 = Block(1,75)
        self.block2 = Block(75,50,dropout = 0.2)
        self.block3 = Block(50,25)
        self.fullyConnectedLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*3*25,512),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(512,n_class)
        )
    def forward(self,x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        output = self.fullyConnectedLayer(block3)
        return output