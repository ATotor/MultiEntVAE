# -*- coding: utf-8 -*-

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def MNIST_give_dataset(root='.'):
    training_data = datasets.MNIST(
        root+'/data',
        train=True,
        download=True,
        transform=ToTensor()
        )
    
    test_data = datasets.MNIST(
        root+'/data',
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
