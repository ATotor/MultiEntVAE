import numpy as np
import os
import torch
import torch.nn as nn
from datetime import datetime


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
    def __init__(self,type='01'):
        super(Normalize, self).__init__()
        self.type=type
    def forward(self, x):
        if len(x.shape) == 3:
            maxval = x.amax(dim=(1,2)).reshape((-1,1,1))
            minval = x.amin(dim=(1,2)).reshape((-1,1,1))
        else:
            minval = x.min()
            maxval = x.max()
        if self.type=="01":
            normalized = (x - minval) / (maxval - minval)
        if self.type=='max1':
            normalized = x/np.abs(x).max()
        return normalized
    
class func2module(nn.Module):
    def __init__(self,f):
        super(func2module,self).__init__()
        self.f = f
    def forward(self,x):
        return self.f(x)

