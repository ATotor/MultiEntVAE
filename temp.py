# -*- coding: utf-8 -*-

import os
import torch
from src.dataset_loader import *
from src.VAE import *

import matplotlib.pyplot as plt

print(os.getcwd())
root = os.getcwd()
batch_size = 64
train_dataloader, test_dataloader =  MNIST_give_dataloader(root, batch_size)

model = VAE(10)

epochs = 5
lr = 1e-3
model = train_VAE(model, train_dataloader, epochs, lr)

x, _ = next(iter(train_dataloader))
xbar, kldiv = model(x)

with torch.no_grad():
    plt.imshow(x[0].reshape(28,28), cmap="gray")
    plt.figure()
    plt.imshow(xbar[0].reshape(28,28), cmap="gray")