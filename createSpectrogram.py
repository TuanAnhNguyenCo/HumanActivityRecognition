import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft
import torch
import glob
from einops import rearrange

def compute_spectrogram(data, fs=1.0, nperseg=256, noverlap=None):
    """Compute the spectrogram using Short-Time Fourier Transform (STFT)"""
    frequencies, times, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return frequencies, times, Zxx

# data = torch.load("data/new_data/emg_test.pkl")

# for url in data:
#     # emg,label  = torch.load("data/new_data/emg_data/PhongTrang_20230701_02_0_0.pkl")
#     emg,label  = torch.load(url)
#     new_url = url.replace("emg_data","spectrogram")
    
#     emg = np.array(emg).T

#     Z = []
#     for i in range(8):
#         _,_,Zxx = compute_spectrogram(emg[i])
#         Z.append(Zxx)
#     Z = np.array(Z)
#     torch.save((Z,label),new_url)

data = glob.glob("data/new_data/spectrogram/*")

for url in data:
    spectrogram, label = torch.load(url.replace("emg_data","spectrogram"))
    if spectrogram.shape[1] == 130:
        torch.save((spectrogram,label),url.replace("spectrogram","new_spectrogram"))
        print("Hix")
        continue
    zeros = np.zeros((8,1,70))
    spectrogram = np.concatenate((abs(spectrogram),zeros),axis = 1)
    torch.save((spectrogram,label),url.replace("spectrogram","new_spectrogram"))


