import pandas as pd
import glob
from preprocessing import read_class_from_file
import numpy as np
import torch


def extract_frame(text, key):
    idx = text.find(key)
    return int(text[idx+len(key)])


def merge_EMG_CSV_file(url, class_name, user_id):  # test and val are 2s
    classes = read_class_from_file()
    data = []
    cnt = 0
    f = 44100

    file_list = glob.glob(url + f"/{class_name}_*{user_id}.csv")
    if file_list != []:
        col = pd.DataFrame()
        sorted_files = sorted(
            file_list, key=lambda x: extract_frame(x, class_name+"_"))
        for file in sorted_files:
            emg_data = pd.read_csv(file)
            col = pd.concat([col, emg_data], axis=1, ignore_index=True)

        col = col.iloc[:f*3, :]
        col = col.to_numpy()
        return torch.tensor(col)
    return None
