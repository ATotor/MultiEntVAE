# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

def disp_loss(loss):
    with torch.no_grad():
        plt.figure(figsize=(15,5))
        plt.plot(loss)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        
def disp_MNIST_example(model, dataloader):
    x, _ = next(iter(dataloader))
    xbar = model(x)
    fig, ax = plt.subplots(2,5)
    with torch.no_grad():
        for i in range(5):
            ax[0,i].imshow(x[i].reshape(28,28), cmap="gray")
            ax[1,i].imshow(xbar[i].reshape(28,28), cmap="gray")
    plt.show()
    
    
def disp_MNIST_latent(model, dims=(4,4), n_channels=12):
    fig, ax = plt.subplots(dims[0],dims[1])
    for i in range(dims[0]):
        for j in range(dims[1]):
            z = torch.zeros((1,n_channels,dims[0],dims[1]))
            z[0,:,i,j] = 1
            y = model.decode(z)
            with torch.no_grad():
                ax[i, j].imshow(y.reshape(28, 28), cmap="gray")
    plt.show()
    
    
def disp_MNIST_img(model, dataloader, writer):
    x,_ = next(iter(dataloader))
    
    grid = torchvision.utils.make_grid(x)
    writer.add_image('images/ground_truth', grid)
    grid = torchvision.utils.make_grid(model(x))
    writer.add_image('images/model', grid)
    

def disp_test():
    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer.close()