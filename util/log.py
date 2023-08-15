from datetime import datetime
import os
import torch
import pandas as pd

class Log:
    def __init__(self,root,model_name):
        self.root = root
        self.model_name = model_name
        
    def save_model(self,model):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d")
        filename = self.model_name + '@' + current_time + "-weight" + ".pth"
        weight_file = os.path.join(self.root,filename)
        torch.save(model.state_dict(),weight_file)
    
    def save_training_loss(self,train_losses):
        df_train_loss = pd.DataFrame(train_losses)
        df_train_loss.to_csv(os.path.join(self.root,"train_loss.csv"))
    
    def save_training_performance(self,train_log):
        df_train_log = pd.DataFrame(train_log)
        df_train_log.to_csv(os.path.join(self.root,"training_perfomance.csv"))
        
    def save_testing_performance(self,test_log):
        df_test_log = pd.DataFrame(test_log)
        df_test_log.to_csv(os.path.join(self.root,"testing_perfomance.csv"))