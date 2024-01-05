import numpy as np
import torch.nn as nn
import os
import torch
from datetime import datetime
import librosa as li

def next_power_of_2(n):
    # Check if n is already a power of 2
    if n and not (n & (n - 1)):
        return n
    # Find the most significant bit position and shift left
    p = 1
    while p < n:
        p <<= 1
    return p

def find_most_recent_VAE():
    found_files = []
    for _, _, files in os.walk("results"):
        for file in files:
            if len(file)>4 and file[:4] == "VAE_":
                strdate = file[4:]
                date_object = datetime.strptime(strdate,"%d-%m-%Y_%H_%M")
                found_files.append(date_object)
    if found_files:
        found_date = (sorted(found_files)[-1]).strftime("%d-%m-%Y_%H_%M")
        most_recent_file = "VAE_"+found_date
        return most_recent_file
    else:
        raise Exception("No model found")

def load_model(file_name):
    return torch.load('results/'+file_name)
    

def save_model(model):
    with torch.no_grad():
        date_time = datetime.now().strftime("%d-%m-%Y_%H_%M")
        torch.save(model, 
                   os.path.join("results","VAE_"+date_time))
def save_loss(loss):
    date_time = datetime.now().strftime("%d-%m-%Y_%H_%M")
    np.save(os.path.join("results","loss_"+date_time), loss)

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self, x):
        minval = x.min()
        maxval = x.max()
        return (x - minval) / (maxval - minval)
    
class librosa_GriffinLim(nn.Module):
    def __init__(self,n_fft=2048):
        super(librosa_GriffinLim, self).__init__()
        self.n_fft = n_fft
    def forward(self, x):
        return li.griffinlim(x.cpu().numpy(),n_fft=self.n_fft)