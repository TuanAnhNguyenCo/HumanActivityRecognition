from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import numpy as np
import torch
import random
from util.spec import _create_spectrogram
from einops import rearrange
from scipy import signal
import matplotlib.pyplot as plt
from ssqueezepy import cwt
import cv2
from torchvision import transforms


class HandSignData(Dataset):
    def __init__(self, img_dir, img_labels, transform=None):
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir[idx]).convert("L")
        label = self.img_labels[idx]
        if self.transform:
            img = self.transform(img)
        img = img/255  # scale image
        return (img, label)


class SkeletonData(Dataset):
    def __init__(self, url):
        self.features, self.labels = torch.load(url)
        print(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])


class MultiModalData(Dataset):
    def __init__(self, url):
        self.data = torch.load(url)
       
        print(len(self.data))
       
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        # try:
            bones = torch.tensor(torch.load(self.data[idx][0]))
            emg, label = torch.load(self.data[idx][1])
            return (bones, torch.tensor(label), torch.tensor(emg))


class MultiModalData1(Dataset):
    def __init__(self, url):
        self.data = url
        # self.data = self.data[:1000]
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        scalogram, label = torch.load(self.data[idx])
        return (0, torch.tensor(label), scalogram)


class SkeletonAndEMGData(Dataset):
    def __init__(self, data):
        self.data = data
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # original emg, normalized emg, label
        emg, _, label = torch.load(self.data[idx])
        while emg.shape[-1] < 8:
            emg, _, label = torch.load(
                self.data[random.randint(0, self.__len__()-1)])
        # return _, label, emg
        emg = rearrange(emg, "a b  -> b a")  # 8*n
        emg = torch.abs(emg)

        return _, label, emg


# (tensor([[0.1758],
#          [0.0866],
#          [0.1913],
#          [0.1871],
#          [0.1908],
#          [0.1897],
#          [0.1893],
#          [0.1857]], dtype=torch.float64),
#  tensor([[-0.1183],
#          [-0.0776],
#          [-0.2787],
#          [-0.2766],
#          [-0.2817],
#          [-0.2794],
#          [-0.2791],
#          [-0.2786]], dtype=torch.float64))
class SkeletonAndEMGData1(Dataset):
    def __init__(self, data):
        self.data = data
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # original emg, normalized emg, label
        emg, label = torch.load(self.data[idx])
        while emg.shape[-1] < 8:
            emg,label = torch.load(
                self.data[random.randint(0, self.__len__()-1)])
        # return _, label, emg
        emg = torch.tensor(rearrange(emg, "a b  -> b a"))  # 8*n
        # ma = torch.tensor([[0.1758],[0.0866],[0.1913],[0.1871],[0.1908],[0.1897],[0.1893],[0.1857]]).reshape(8,1)
        # mi = torch.tensor([[-0.1183],[-0.0776],[-0.2787],[-0.2766],[-0.2817],[-0.2794],[-0.2791],[-0.2786]]).reshape(8,1)
        # emg = 2*(emg - mi)/(ma-mi) - 1
        # emg = abs(emg)
   
        
        


        return 1,label, emg

def load_data(ROOT, train_size, valid_size, test_size, input_dim, n=5):
    folder_name = os.listdir(ROOT)
    labels = {}
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []
    # read and save to X,y
    for i in range(len(folder_name)):
        labels[folder_name[i]] = i
        file_list = glob.glob(os.path.join(ROOT, folder_name[i])+"/*")
        subset_len = len(file_list)//n
        # shuffle
        np.random.shuffle(file_list)

        X_train.extend(file_list[:subset_len*(n-2)])
        X_val.extend(file_list[subset_len*(n-2):subset_len*(n-1)])
        X_test.extend(file_list[subset_len*(n-1):])

        y_train.extend(np.full(len(file_list[:subset_len*(n-2)]), i))
        y_val.extend(
            np.full(len(file_list[subset_len*(n-2):subset_len*(n-1)]), i))
        y_test.extend(np.full(len(file_list[subset_len*(n-1):]), i))

    # convert to tensor (channel, height, width)
    train_transform = transforms.Compose([
        transforms.Resize((input_dim[0], input_dim[0])),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_dim[0], input_dim[0])),
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(HandSignData(
        X_train, y_train, train_transform), batch_size=train_size, drop_last=False)
    valid_loader = DataLoader(HandSignData(
        X_val, y_val, test_transform), batch_size=valid_size, drop_last=False)
    test_loader = DataLoader(HandSignData(
        X_test, y_test, test_transform), batch_size=test_size, drop_last=False)

    return train_loader, valid_loader, test_loader
