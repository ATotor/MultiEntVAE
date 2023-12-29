# -*- coding: utf-8 -*-


import os

#os.chdir('C:/Users/vppit/Desktop/Sorbonne/M_S3/IM/MultiEntVAE')
#os.chdir('../')

import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from src.VAE import *
from src.dataset_loader import *
from src.plots import *


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--disp', type=bool, default=True)
parser.add_argument('--save', type=bool, default=False)
args = parser.parse_args()

epochs = args.epochs
lr = args.lr
beta = args.beta
batch_size = args.batch_size
disp = args.disp
save = args.save

train_dataloader, test_dataloader =  MNIST_give_dataloader(batch_size=batch_size)

model = VAE()

writer = SummaryWriter()

model, loss = train_VAE(model, train_dataloader, writer, epochs, lr, beta)
disp_MNIST_img(model, test_dataloader, writer)
writer.close()

if disp:
    disp_loss(loss)
    disp_MNIST_example(model, test_dataloader)
    
if save:
    save_model_and_loss(model, loss)