from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import glob
import numpy as np
import torch


class HandSignData(Dataset):
    def __init__(self,img_dir,img_labels,transform = None):
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
        img = img/255 # scale image
        return (img,label)

class SkeletonData(Dataset):
    def __init__(self,url):
        self.features,self.labels = load_data_from_file(url)
        print("features = ",len(self.features)," class = ",len(self.labels))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = torch.tensor(feature).reshape(-1,3)
        label = self.labels[idx]
        return (feature,label)

def load_data_from_file(url,is_class = False):
    if not is_class:
        data = []
        labels = []
        with open(url,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = [float(i) for i in line.replace("\n","").split(',')]
                data.append(line[2:])
                labels.append(int(line[1]))
        return data,labels
    else:
        data = {}
        with open(url,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n","").split(':')
                data[int(line[1])] = line[0]
        return data


def load_data_HaGRID(ROOT,n = 5):
   pass
        


    
    
 
def load_data(ROOT,train_size,valid_size,test_size,input_dim,n = 5):
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
        file_list = glob.glob(os.path.join(ROOT,folder_name[i])+"/*")
        subset_len = len(file_list)//n
        #shuffle
        np.random.shuffle(file_list)
        
        X_train.extend(file_list[:subset_len*(n-2)])
        X_val.extend(file_list[subset_len*(n-2):subset_len*(n-1)])
        X_test.extend(file_list[subset_len*(n-1):])
        
        y_train.extend(np.full(len(file_list[:subset_len*(n-2)]),i))
        y_val.extend(np.full(len(file_list[subset_len*(n-2):subset_len*(n-1)]),i))
        y_test.extend(np.full(len(file_list[subset_len*(n-1):]),i))
    
    # convert to tensor (channel, height, width)
    train_transform = transforms.Compose([
        transforms.Resize((input_dim[0],input_dim[0])),
        transforms.ToTensor(),  
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_dim[0],input_dim[0])),
        transforms.ToTensor(),  
    ])
    train_loader = DataLoader(HandSignData(X_train,y_train,train_transform),batch_size = train_size,drop_last = False)
    valid_loader = DataLoader(HandSignData(X_val,y_val,test_transform),batch_size = valid_size,drop_last = False)
    test_loader = DataLoader(HandSignData(X_test,y_test,test_transform),batch_size = test_size,drop_last = False)  
   
    return train_loader,valid_loader,test_loader