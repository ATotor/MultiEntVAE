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

def NSYNTH_give_dataset(root=".",device=torch.device("cpu"),transform = nn.Identity()):
    training_data = NSynth(  
                top_path = root , 
                device = device,
                n_signal = 32000,
                valid_pitch = None, 
                valid_inst = None, 
                valid_source= None,
                transform = transform,
    )
    test_data = training_data #A CHANGER
    
    return training_data, test_data

def NSYNTH_give_dataloader(root='.',batch_size = 64,device=torch.device("cpu"),transform= nn.Identity()):
    training_data, test_data = NSYNTH_give_dataset(root,device=device,transform=transform)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    train_spec_normalizer, train_spec_denormalizer = find_spec_normalizer(train_dataloader)
    test_spec_normalizer, test_spec_denormalizer = find_spec_normalizer(train_dataloader)

    return train_dataloader, test_dataloader, train_spec_normalizer, train_spec_denormalizer, test_spec_normalizer, test_spec_denormalizer