import argparse 
from util.conf import *
from loader.dataloader import load_data
from model.HandSignCNNModel import HandSignCNNModel
from base.trainer import Trainer
from torch import nn
import torch

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
    parser.add_argument('--epochs', type=float, default=50,
                        help='seed')
    parser.add_argument('--train_batch_size', type=float, default=256,
                        help='seed')
    parser.add_argument('--val_batch_size', type=float, default=64,
                        help='seed')
    parser.add_argument('--test_batch_size', type=float, default=64,
                        help='seed')
    parser.add_argument('--loss_function', type = str , default="CrossEntropyLoss",choices=[""],
                        help='seed')
    parser.add_argument('--optimizer', type = str , default="SGD",choices=["Adam"],
                        help='seed')
    parser.add_argument('--device', type = str , default="cuda:0",choices=["cpu"],
                        help='seed')
    parser.add_argument('--n_classes', type = int , default=18,
                        help='seed')
    parser.add_argument('--root', type = str , default="./data/image",
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
    main()
    print("End")