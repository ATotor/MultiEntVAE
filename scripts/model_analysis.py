# -*- coding: utf-8 -*-

import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from src.VAE import *
from src.dataset_loader import *
from src.plots import *


parser = ArgumentParser()
parser.add_argument('--filename', type=str, default='VAE_23-12-2023_18_30')
args = parser.parse_args()
file_name = args.filename

model = load_model(file_name)
_, test_dataloader =  MNIST_give_dataloader()

disp = True
while disp:
    disp_MNIST_example(model, test_dataloader)
    disp = False
    ans = input('Continue (y/n) ? ')
    if ans=='y':
        disp = True