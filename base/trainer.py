from tqdm.auto import tqdm
from torch import nn
from loader.dataloader import load_data
from loader.criterion_loader import load_criterion
from loader.model_loader import load_model
from loader.optimizer_loader import load_optimizer
from util.accuracy_torch import get_accuracy
from util.statistical_results import plot_accuracy,plot_losses
from util.log import Log 
from util.evaluation import early_stopping
import os
import torch


class Trainer:
    def __init__(self,args):
        self.train_loader, self.valid_loader, self.test_loader = load_data(
            args['root'],args['train_batch_size'],
            args['val_batch_size'],args['test_batch_size'],args['input_dim']
        )
        self.device = args['device']
        self.epochs = args['epochs']
        self.model = load_model(args['model'],args['n_classes']).to(self.device)
        self.criterion = load_criterion(args['loss_function'])
        self.optimizer = load_optimizer(args['optimizer'],self.model,args['lrate'])
        self.output = f"./log/{args['dataset']}/@{args['model']}-inp_emb:{args['input_dim'][0]}x{args['input_dim'][1]}-lr:{args['lrate']}-ep:{args['epochs']}-l:{args['loss_function']}-op:{args['optimizer']}-wdecay:{args['weight_decay']}"
        self.log = Log(self.output,args['model'])
        
        if not os.path.exists(self.output):
            os.makedirs(self.output)
    
        
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
        test_accuracy = get_accuracy(self.model,self.test_loader,self.device)
        print("Test set accuracy = ",test_accuracy)
        self.log.save_testing_performance(
            {
                "test_accuracy":[test_accuracy]
            }
        )
        print("Save log successfully")
       
        
    
    def save_log(self,train_losses,valid_losses,train_accuracy,val_accuracy):
        self.log.save_training_loss({
            "Epoch":range(self.epochs),
            "Training loss": train_losses,
            "Validation loss":valid_losses
        })
        self.log.save_training_performance({
            "Epoch":range(self.epochs),
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy
        })
        print("Save log successfully")
        
    
    def train(self):
        train_losses = []
        valid_losses = []
        train_accuracy = []
        val_accuracy = []
        
        max_val_accuracy = -1

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
            
            # save best model
            if val_acc > max_val_accuracy:
                max_val_accuracy = val_acc
                self.log.save_model(self.model)
                print("Save best model")

            # early stopping
            _,should_stop= early_stopping(val_accuracy,stopping_steps=7)
            if should_stop:
                break
        
        # save log
        self.save_log(train_losses,valid_losses,train_accuracy,val_accuracy)
        
        # plot results
        plot_losses(train_losses,valid_losses)
        plot_accuracy(train_accuracy,val_accuracy)
        
        
        
        
         