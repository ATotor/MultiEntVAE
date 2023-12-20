# -*- coding: utf-8 -*-

import os
import torch
from src.dataset_loader import *
from src.VAE import *

print(os.getcwd())
root = os.getcwd()
batch_size = 64
train_dataloader, test_dataloader =  MNIST_give_dataloader(root, batch_size)

model = AE

x, y = next(iter(train_dataloader))
xbar = model(x)