# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision


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

def tensorboard_writer(model, dataloader,writer,device):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images.to(device)
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)
        writer.close()
