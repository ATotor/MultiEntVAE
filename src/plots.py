# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
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

def tensorboard_writer(model:nn.Module, 
                       dataloader:torch.utils.data.Dataset,
                       writer:SummaryWriter,
                       inverse_transform:nn.Module,
                       normalizer:nn.Module,
                       batch_size:int|None=None,
                       epoch:int|str = "Final"
                       ):
    if epoch=="Final":
        print("Creating logs")
    model.eval()
    with torch.no_grad():
        item = next(iter(dataloader))
        spec = item['x']
        spec = normalizer(spec)
        batch_size = spec.shape[0] if batch_size is None else batch_size

        images = spec[:,None,:,:]
        output, _ = model(spec)
        output_image = output[:,None,:,:]
        
        torch.save(spec,'original_spec_tensor')
        torch.save(output,'generated_spec_tensor')
        
        for batch_number in range(batch_size):
            init_audio = inverse_transform(spec[batch_number])
            gen_audio = inverse_transform(output[batch_number])
            original_audio = item["audio"][batch_number]
            if batch_number == 0:
                for i in range(0,len(gen_audio),300): 
                    writer.add_scalar(f"Batch number {batch_number} Epoch {epoch}/Generated Waveform",gen_audio[i],i)
                    writer.add_scalar(f"Batch number {batch_number} Epoch {epoch}/Initial recreated Waveform",init_audio[i],i)
                for i in range(0,len(original_audio),300):
                    writer.add_scalar(f"Batch number {batch_number} Epoch {epoch}/Initial Waveform",original_audio[i],i)

            writer.add_image(f'Batch number {batch_number} Epoch {epoch}/Initial Spectrogram {item["fname"][batch_number]}',images[batch_number])
            writer.add_image(f'Batch number {batch_number} Epoch {epoch}/Generated Spectrogram {item["fname"][batch_number]}',output_image[batch_number])
            writer.add_audio(f'Batch number {batch_number} Epoch {epoch}/Initial audio {item["fname"][batch_number]}',original_audio,sample_rate = 16000)
            writer.add_audio(f'Batch number {batch_number} Epoch {epoch}/Initial recreated audio {item["fname"][batch_number]}',init_audio,sample_rate = 16000)
            writer.add_audio(f'Batch number {batch_number} Epoch {epoch}/Generated audio {item["fname"][batch_number]}',gen_audio,sample_rate = 16000)

        writer.close()


def log_model_grad_norm(model: nn.Module, tb: SummaryWriter, step: int):
    norms = torch.stack([torch.norm(p.grad.detach(), p=2.0) for p in model.parameters() if p.requires_grad])
    tb.add_scalar("grad_norm/norm", norms.norm(2.0), step)
    tb.add_scalar("grad_norm/mean", norms.mean(), step)
    return

def log_model_loss(writer: SummaryWriter, full_loss: torch.Tensor, full_mse: torch.Tensor, full_kl: torch.Tensor, epoch: int):
    writer.add_scalar("Loss/total loss", full_loss.item(), epoch) 
    writer.add_scalar("Loss/reconstruction loss", full_mse.item(), epoch) 
    writer.add_scalar("Loss/kl div", full_kl.item(), epoch) 
    return

def log_arg(writer,args):
    writer.add_text(f"beta",str(args.beta))
    writer.add_text(f"learning rate",str(args.lr))
    return