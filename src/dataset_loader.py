# -*- coding: utf-8 -*-

import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
from src.dataset import *
from src.utils import *
from src.preprocessing import *


def MNIST_give_dataset(root='.'):
    training_data = datasets.MNIST(
        os.path.join(root,'data'),
        train=True,
        download=True,
        transform=ToTensor()
        )
    
    test_data = datasets.MNIST(
        os.path.join(root,'data'),
        train=False,
        download=True,
        transform=ToTensor()
        )
    
    return training_data, test_data


def MNIST_give_dataloader(root='.', batch_size=64):
    training_data, test_data = MNIST_give_dataset(root)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

def NSYNTH_give_dataset(training_path=".",testing_path=".",device=torch.device("cpu"),transform = nn.Identity()):
    training_data = NSynth(  
                top_path = training_path , 
                device = device,
                n_signal = 2,
                valid_pitch = [24, 96], 
                valid_inst = None, 
                valid_source= ["acoustic"],
                transform = transform,
    )
    test_data = NSynth(  
                top_path = testing_path , 
                device = device,
                n_signal = 2,
                valid_pitch = [24, 96], 
                valid_inst = None, 
                valid_source= ["acoustic"],
                transform = transform,
    )
    
    return training_data, test_data

def NSYNTH_give_dataloader(training_path=".",testing_path=".",batch_size = 64,device=torch.device("cpu"),transform= nn.Identity()):
    training_data, test_data = NSYNTH_give_dataset(training_path,testing_path,device=device,transform=transform)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    return train_dataloader, test_dataloader