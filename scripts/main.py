# -*- coding: utf-8 -*-


import os

#os.chdir('C:/Users/vppit/Desktop/Sorbonne/M_S3/IM/MultiEntVAE')
#os.chdir('../')

import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from src.VAE import *
from src.dataset_loader import *


parser = ArgumentParser()
args = parser.parse_args()
try :
    epochs = float(args.epochs)
except:
    epochs = 30
try :
    lr = float(args.lr)
except:
    lr = 1e-3
try:
    batch_size = args.batch_size
except:
    batch_size = 64
try:
    disp = bool(args.disp)
except:
    disp = False
try:
    save = bool(args.save)
except:
    save = False

root = os.getcwd()

train_dataloader, test_dataloader =  MNIST_give_dataloader(root, batch_size)

model = VAE(10)

model, loss = train_VAE(model, train_dataloader, epochs, lr)    

if disp:
    x, _ = next(iter(train_dataloader))
    xbar, kldiv = model(x)
    with torch.no_grad():
        plt.imshow(x[0].reshape(28,28), cmap="gray")
        plt.figure()
        plt.imshow(xbar[0].reshape(28,28), cmap="gray")

if disp:
    disp_loss(loss)
    
if save:
    save_model_and_loss(model, loss)