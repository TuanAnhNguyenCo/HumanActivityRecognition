from tqdm.auto import tqdm
import torch
from util.accuracy_torch import get_accuracy
from util.statistical_results import plot_accuracy,plot_losses

class Trainer:
    def __init__(self,train_loader,valid_loader,test_loader,
                 model,criterion,optimizer,device,epochs         
                 ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
    
        
    def train_each_epoch(self):
        running_loss = 0
        self.model.train()
        for images,labels in tqdm(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # forward
            outputs = self.model(images)
            loss = self.criterion(outputs,labels)
            running_loss += loss.item()
            
            #backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
            
    def evaluate(self):
        self.model.eval()
        running_loss = 0
        for images,labels in tqdm(self.valid_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # forward
            outputs = self.model(images)
            loss = self.criterion(outputs,labels)
            running_loss += loss.item()

        epoch_loss = running_loss / len(self.valid_loader)
        
        return epoch_loss
    
    def test(self):
        print("Test set accuracy = ",get_accuracy(self.model,self.test_loader,self.device))
    
    def train(self):
        train_losses = []
        valid_losses = []
        train_accuracy = []
        val_accuracy = []

        for epoch in range(self.epochs):
            # training
            train_loss= self.train_each_epoch()
            
            # validation
            with torch.no_grad():
                valid_loss = self.evaluate()
            train_acc = get_accuracy(self.model,self.train_loader,self.device)
            val_acc = get_accuracy(self.model,self.valid_loader,self.device)
            print("Epoch {} --- Train loss = {} --- Valid loss = {} -- Train set accuracy = {} % Valid set Accuracy = {} %".format
                (epoch+1,train_loss,valid_loss,train_acc,val_acc))
            # save loss value
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            #save accuracy
            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)
        
        plot_losses(train_losses,valid_losses)
        plot_accuracy(train_accuracy,val_accuracy)
        
        
         