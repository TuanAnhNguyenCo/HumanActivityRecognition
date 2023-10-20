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
        # self.data = self.data[:1000]
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            bones = torch.tensor(torch.load(self.data[idx][0]))
            _, emg, label = torch.load(self.data[idx][1])
            while emg.shape[-1] < 8 or bones.shape[1] < 21:
                new_id = random.randint(0, self.__len__()-1)
                _, emg, label = torch.load(
                    self.data[new_id][1])
                bones = torch.tensor(torch.load(self.data[new_id][0]))
        except:
            idx = random.randint(0, self.__len__()-1)
            bones = torch.tensor(torch.load(self.data[idx][0]))
            _, emg, label = torch.load(self.data[idx][1])
            while emg.shape[-1] < 8 or bones.shape[1] < 21:
                new_id = random.randint(0, self.__len__()-1)
                _, emg, label = torch.load(
                    self.data[new_id][1])
                bones = torch.tensor(torch.load(self.data[new_id][0]))

        return (bones, torch.tensor(label), emg)


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
        _, emg, label = torch.load(self.data[idx])
        while emg.shape[-1] < 8:
            _, emg, label = torch.load(
                self.data[random.randint(0, self.__len__()-1)])
        # return _, label, emg
        return _, label, rearrange(emg, "a b  -> b a") # 8*n


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
