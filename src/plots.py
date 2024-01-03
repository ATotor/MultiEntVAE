# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchaudio import transforms as T
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

def tensorboard_writer(model, dataloader,writer,inverse_transform,device):
    print("Creating logs")
    model.eval()
    with torch.no_grad():
        item = next(iter(dataloader))
        spec = item['x']
        batch_size = spec.shape[0]
        init_audio = item['audio']
        images = spec[:,None,:,:]
        
        output = model(spec)
        output_image = output[:,None,:,:]
        gen_audio = inverse_transform(output)

        for batch_number in range(batch_size):
            writer.add_image(f'{batch_number}/Initial Spectrogram',images[batch_number])
            writer.add_image(f'{batch_number}/Generated Spectrogram',output_image[batch_number])

            writer.add_audio(f"{batch_number}/Initial audio",init_audio[batch_number],sample_rate = 16000)

    
            writer.add_audio(f"{batch_number}/Generated audio",gen_audio[batch_number],sample_rate = 16000)

            for i in range(len(gen_audio[batch_number])): 
                writer.add_scalar(f"{batch_number}/Generated Waveform",gen_audio[batch_number][i],i)
                writer.add_scalar(f"{batch_number}/Initial Waveform",init_audio[batch_number][i],i)

        writer.close()
