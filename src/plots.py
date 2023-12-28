# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from librosa import griffinlim
from soundfile import write

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
    print("Creating logs")
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images.to(device)
        output = model(images)
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)

        writer.add_image('Initial Spectrogram',images[0])
        writer.add_image('Generated Spectrogram',output[0])

        init_audio = griffinlim(images[0].cpu().numpy(),n_fft=2054,hop_length=472)
        init_audio = torch.Tensor(init_audio)
        writer.add_audio("Initial audio",init_audio,sample_rate = 16000)

        
        gen_audio = griffinlim(output[0].cpu().numpy(),n_fft=2054,hop_length=472)
        gen_audio = torch.Tensor(gen_audio)
        writer.add_audio("Generated audio",gen_audio,sample_rate = 16000)


        for i in range(len(gen_audio[0])): 
            writer.add_scalar("Generated Waveform",gen_audio[0][i],i)
            writer.add_scalar("Initial Waveform",init_audio[0][i],i)
        writer.close()
