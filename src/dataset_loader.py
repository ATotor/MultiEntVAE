# -*- coding: utf-8 -*-

import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchaudio.transforms import Spectrogram

from src.dataset import NsynthDataset


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

def NSYNTH_give_dataset(root='.'):
    training_data = NsynthDataset(  
            json_file = os.path.join(root,"data","nsynth-test","examples.json"),
            sound_dir = os.path.join(root,"data","nsynth-test","audio") ,
            transform = Spectrogram(n_fft=2054,hop_length=488) )
    test_data = training_data #A CHANGER
    
    return training_data, test_data

def NSYNTH_give_dataloader(root='.',batch_size = 64):
    training_data, test_data = NSYNTH_give_dataset(root)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader