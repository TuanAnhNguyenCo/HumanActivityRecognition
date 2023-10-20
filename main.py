import argparse 
from util.conf import *
from loader.dataloader import load_data
from model.HandSignCNNModel import HandSignCNNModel
from base.trainer import Trainer
from torch import nn
import torch
import numpy as np
from util.img2bone import HandDetector
import torch.nn.functional as F

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the argumentss
    parser.add_argument('--model', type=str, default='CNN',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='SubsampleHAR', choices=[''],
                        help='Dataset name')
    parser.add_argument('--lrate', type=float, default=0.01,
                        help='seed')
    parser.add_argument('--epochs', type=float, default=3,
                        help='seed')
    parser.add_argument('--train_batch_size', type=float, default=128,
                        help='seed')
    parser.add_argument('--val_batch_size', type=float, default=64,
                        help='seed')
    parser.add_argument('--test_batch_size', type=float, default=64,
                        help='seed')
    parser.add_argument('--loss_function', type = str , default="CrossEntropyLoss",choices=[""],
                        help='seed')
    parser.add_argument('--optimizer', type = str , default="SGD",choices=["Adam"],
                        help='seed')
    parser.add_argument('--device', type = str , default="cuda:1",choices=["cpu"],
                        help='seed')
    parser.add_argument('--n_classes', type = int , default=18,
                        help='seed')
    parser.add_argument('--root', type = str , default="./data/image/subsampleHAR",
                        help='seed')
    parser.add_argument('--input_dim', type = list , default=[28,28],
                        help='seed')
    parser.add_argument('--weight_decay', type = int , default=0,
                        help='seed')
    return parser.parse_args()

def ModelLoader(args):
    
    model = Trainer(args)
        
    return model 

def main():
    args = parse_arguments()
    args = namespace_to_dict(args)
    model = ModelLoader(args)
    model.train()
    model.test()
    
if __name__ == "__main__":
    print("Run")
    # main()
    # handDetector = HandDetector()
    # handDetector.findHands("data/image/subsampleHAR/dislike/001c6f56-85cf-4e45-bfc1-1af53c0e501b.jpg")
    terminal = nn.Parameter(torch.randn(5, 1, 13))
    head_la = F.interpolate(torch.stack([terminal[0],terminal[1]],2), 6)
    head_ra = F.interpolate(torch.stack([terminal[0],terminal[2]],2), 6)
    lw_ra = F.interpolate(torch.stack([terminal[3],terminal[4]],2), 6)
    node_features = torch.cat([
								   (head_la[:,:,:3] + head_ra[:,:,:3])/2,
								   torch.stack((lw_ra[:,:,2], lw_ra[:,:,1], lw_ra[:,:,0]), 2),
								   lw_ra[:,:,3:], head_la[:,:,3:], head_ra[:,:,3:]], 2)
    print(node_features.shape)
    print("End")