# -*- coding: utf-8 -*-

import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from src.VAE import *
from src.dataset_loader import *
from src.plots import *


parser = ArgumentParser()
parser.add_argument('--filename', type=str, default='VAE_24-12-2023_10_49')
parser.add_argument('--plot', type=str, default='latent')
args = parser.parse_args()
file_name = args.filename
plot = args.plot

model = load_model(file_name)
_, test_dataloader =  MNIST_give_dataloader()


disp = True
while disp:
    if plot=='latent':
        disp_MNIST_latent(model)
    elif plot=='example':
        disp_MNIST_example(model, test_dataloader)
    elif plot=='tensorboard':
        disp_MNIST_img(model, test_dataloader)
    else:
        disp_test()
        
    disp = False
    ans = input('Continue (y/n) ? ')
    if ans=='y':
        disp = True
