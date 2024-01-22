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
        
def disp_MNIST_example(model, dataloader,conditional=True):
    x,y= next(iter(dataloader))
    if conditional==False:
        xbar = model(x)
    else:
        xbar_unlab=model(x)
        xbar = model(x,y)
    fig, ax = plt.subplots(2,5)
    with torch.no_grad():
        if conditional==False:
            for i in range(5):
                ax[0,i].imshow(x[i,0,:,:], cmap="gray")
                ax[0,i].set_title(f'In{i},lab={y[i].item()}')
                ax[1,i].imshow(xbar[0][i,0,:,:], cmap="gray")
                ax[1,i].set_title(f'Out{i},lab={y[i].item()}')
        else:
            for i in range(5):
                ax[0,i].imshow(xbar[0][i,0,:,:], cmap="gray")
                ax[0,i].set_title(f'lab={y[i].item()},IN with')
                ax[1,i].imshow(xbar_unlab[0][i,0,:,:], cmap="gray")
                ax[1,i].set_title(f'lab={y[i].item()},IN without')
    return(fig)
def disp_Mnist_random_sample(model,H,W,label=None):
    X=torch.randn(H*W,model.latent_dims)
    fig, ax = plt.subplots(H,W)
    Y=model.decode(X)   
    with torch.no_grad():
        for i in range(H):
            for j in range(W):    
                ax[i,j].imshow(Y[i+j*W,0,:,:], cmap="gray")        
    plt.show()
    return(fig)

def disp_Mnist_random_sample_condition(model,H,W,label):
    X=torch.randn(H*W,model.latent_dims)#lattent_dim
    y=torch.ones(H*W,)*label
    z=model.condition_on_label(X, y)
    Z=model.decode(z)
    fig, ax = plt.subplots(H,W)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                ax[0,i].set_title(f"In{i+j*W},lab={label}")
                ax[i,j].imshow(Z[i+j*W,0,:,:], cmap="gray")        
    plt.show()
    return(fig)
    
    Y=model.decode(X)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):    
                ax[i,j].imshow(Y[i+j*W,0,:,:], cmap="gray")        
    plt.show()
    return(fig)
   
def tensorboard_writer(model, dataloader,writer,inverse_transform,train_spec_normalizer,args):
    print("Creating logs")
    writer.add_text(f"beta",str(args.beta))
    writer.add_text(f"learning rate",str(args.lr))
    model.eval()
    with torch.no_grad():
        item = next(iter(dataloader))
        spec = item['x']
        spec = train_spec_normalizer(spec)
        batch_size = spec.shape[0]

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
                    writer.add_scalar(f"{batch_number}/Generated Waveform",gen_audio[i],i)
                    writer.add_scalar(f"{batch_number}/Initial recreated Waveform",init_audio[i],i)
                for i in range(0,len(original_audio),300):
                    writer.add_scalar(f"{batch_number}/Initial Waveform",original_audio[i],i)

            writer.add_image(f'{batch_number}/Initial Spectrogram {item["fname"][batch_number]}',images[batch_number])
            writer.add_image(f'{batch_number}/Generated Spectrogram {item["fname"][batch_number]}',output_image[batch_number])
            writer.add_audio(f'{batch_number}/Initial audio {item["fname"][batch_number]}',original_audio,sample_rate = 16000)
            writer.add_audio(f'{batch_number}/Initial recreated audio {item["fname"][batch_number]}',init_audio,sample_rate = 16000)
            writer.add_audio(f'{batch_number}/Generated audio {item["fname"][batch_number]}',gen_audio,sample_rate = 16000)

        writer.close()


def log_model_grad_norm(model: nn.Module, tb: SummaryWriter, step: int):
    norms = torch.stack([torch.norm(p.grad.detach(), p=2.0) for p in model.parameters() if p.requires_grad])
    tb.add_scalar("grad_norm/norm", norms.norm(2.0), step)
    tb.add_scalar("grad_norm/mean", norms.mean(), step)
    return

def log_model_loss(writer: SummaryWriter, full_loss: torch.Tensor,full_loss_val: torch.Tensor, full_mse: torch.Tensor,full_mse_val: torch.Tensor, full_kl: torch.Tensor,full_kl_val: torch.Tensor, epoch: int):
    writer.add_scalar("Loss/total loss", full_loss.item(), epoch) 
    writer.add_scalar("Loss/reconstruction loss", full_mse.item(), epoch) 
    writer.add_scalar("Loss/kl div ", full_kl.item(), epoch) 
    writer.add_scalar("Loss/total loss val", full_loss_val.item(), epoch) 
    writer.add_scalar("Loss/reconstruction loss val", full_mse_val.item(), epoch) 
    writer.add_scalar("Loss/kl div val", full_kl_val.item(), epoch) 
    return