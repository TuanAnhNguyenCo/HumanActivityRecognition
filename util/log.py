from datetime import datetime
import os
import torch
import pandas as pd


class Log:
    def __init__(self, root, model_name):
        self.root = root
        self.model_name = model_name

    def save_model(self, model):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d")
        filename = self.model_name + '@' + current_time + "-weight" + ".pth"
        weight_file = os.path.join(self.root, filename)
        torch.save(model.state_dict(), weight_file)

    def save_training_log(self, train_losses, train_accuracy, train_f1score_weighted, train_f1score_micro):
       # Create a DataFrame
        data = {
            'train_losses': train_losses,
            'train_accuracy': train_accuracy,
            'train_f1score_weighted': train_f1score_weighted,
            'train_f1score_micro': train_f1score_micro
        }
        df_train_loss = pd.DataFrame(data)
        df_train_loss.to_csv(os.path.join(self.root, "train_log.csv"))

    def save_val_log(self, valid_losses, val_accuracy, val_f1scroe_weighted, val_f1scroe_micro):
        data = {
            'valid_losses': valid_losses,
            'val_accuracy': val_accuracy,
            'val_f1score_weighted': val_f1scroe_weighted,
            'val_f1score_micro': val_f1scroe_micro
        }
        df_val_loss = pd.DataFrame(data)
        df_val_loss.to_csv(os.path.join(self.root, "val_log.csv"))

    def save_test_log(self, test_log):
        test_log = torch.tensor(test_log).reshape(-1, 3).numpy()

        df_test_loss = pd.DataFrame(test_log, columns=[
                                    'test_accuracy', 'test_f1score_weighted', 'test_f1score_micro'])
        df_test_loss.to_csv(os.path.join(self.root, "test_log.csv"))


        