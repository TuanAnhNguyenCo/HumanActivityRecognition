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
    

class ThreeModalData(Dataset):
    def __init__(self, url):
        self.data = torch.load(url)
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # try:
        bones = torch.tensor(torch.load(self.data[idx][0]))
        emg, label = torch.load(self.data[idx][1])
        spectrogram, label = torch.load(self.data[idx][1].replace("emg_data","new_spectrogram"))
        return (bones, torch.tensor(label), torch.tensor(emg),torch.tensor(spectrogram))


class SpectrogramData(Dataset):
    def __init__(self, url):
        self.data = torch.load(url)
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram, label = torch.load(self.data[idx].replace("emg_data","spectrogram"))
        zeros = np.zeros((8,1,70))
        spectrogram = np.concatenate((abs(spectrogram),zeros),axis = 1)
        return (0, torch.tensor(label), torch.tensor(spectrogram))


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
class SkeletonAndEMGData(Dataset):
    def __init__(self, data):
        self.data = data
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # original emg, normalized emg, label
        emg, label = torch.load(self.data[idx])

        # return _, label, emg
        emg = torch.tensor(rearrange(emg, "a b  -> b a"))  # 8*n
        # ma = torch.tensor([[0.1758],[0.0866],[0.1913],[0.1871],[0.1908],[0.1897],[0.1893],[0.1857]]).reshape(8,1)
        # mi = torch.tensor([[-0.1183],[-0.0776],[-0.2787],[-0.2766],[-0.2817],[-0.2794],[-0.2791],[-0.2786]]).reshape(8,1)
        # emg = 2*(emg - mi)/(ma-mi) - 1
        # emg = abs(emg)

        return 1, label, emg



