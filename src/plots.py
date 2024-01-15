# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchaudio import transforms as T
from mpl_toolkits.axes_grid1 import ImageGrid
from src.utils import * 

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
                       item1:dict,
                       item2:dict,
                       writer:SummaryWriter,
                       inverse_transform:nn.Module,
                       normalizer:nn.Module,
                       batch_size:int|None=None,
                       epoch:int=0,
                       dataloader_type:str='Validation',
                       n_steps_interp:int=8
                       ):

    model.eval()
    with torch.no_grad():
        item = item1
        spec = item['x']
        spec = normalizer(spec)
        batch_size = spec.shape[0] if batch_size is None else batch_size

        images = spec[:,None,:,:]
        output, _ = model(spec)
        output_image = output[:,None,:,:]

        init_audio = inverse_transform(spec)
        gen_audio = inverse_transform(output)
        original_audio = item["audio"]

        torch.save(spec,'original_spec_tensor')
        torch.save(output,'generated_spec_tensor')
        
        for batch_number in range(batch_size):
            writer.add_image(f'{dataloader_type} {batch_number}/Initial Spectrogram {item["fname"][batch_number]}',images[batch_number])
            writer.add_image(f'{dataloader_type} {batch_number}/Generated Spectrogram {item["fname"][batch_number]}',output_image[batch_number],global_step=epoch)
            writer.add_audio(f'{dataloader_type} {batch_number}/Initial audio {item["fname"][batch_number]}',original_audio[batch_number],sample_rate = 16000)
            writer.add_audio(f'{dataloader_type} {batch_number}/Initial recreated audio {item["fname"][batch_number]}',init_audio[batch_number],sample_rate = 16000)
            writer.add_audio(f'{dataloader_type} {batch_number}/Generated audio {item["fname"][batch_number]}',gen_audio[batch_number],sample_rate = 16000,global_step=epoch)

        #-----------------------------Random latent space exploration-------------------
        generated = model.generate(device=spec.device)
        generated_audio = inverse_transform(generated)
        gen_batch,*_ = generated_audio.shape
        for i in range(gen_batch):
            audio = generated_audio[i]
            writer.add_image(f'Generated/Spectrogram {i}', generated[:,None,:,:][i],global_step=epoch)
            writer.add_audio(f'Generated/Audio {i}', audio, sample_rate = 16000,global_step=epoch)

        #-----------------------------Interpolation---------------------------
        spec2= item2['x']
        spec2 = normalizer(spec2)
        original_audio2 = inverse_transform(spec2)
        
        interp_spec = interp(model,spec,spec2,n_steps_interp)

        n_col = int(np.ceil(np.sqrt(n_steps_interp+1)))
        for batch_number in range(batch_size):
            generated_interp, _ = model(interp_spec[:,batch_number,:,:])
            generated_audio_interp = inverse_transform(generated_interp)
            fig = plt.figure(figsize=(8., 8.))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(n_col, n_col), 
                            axes_pad=0.5,  # pad between axes in inch.
                            )
            
            audio_cat = torch.tensor([],device=spec.device)
            for i,z in enumerate(zip(grid,generated_interp,generated_audio_interp)):
                ax,generated,generated_audio = z
                ax.imshow(generated.cpu())
                audio_cat=torch.cat([audio_cat,generated_audio])
                ax.set_title(f'{i/n_steps_interp}')

            writer.add_figure(f"{dataloader_type} Interpolation {batch_number}/Spectrograms",fig,global_step=epoch)
            writer.add_audio(f'{dataloader_type} Interpolation {batch_number}/Original Audio 1', init_audio[batch_number], sample_rate = 16000)
            writer.add_audio(f'{dataloader_type} Interpolation {batch_number}/Original Audio 2', original_audio2[batch_number], sample_rate = 16000)
            writer.add_audio(f'{dataloader_type} Interpolation {batch_number}/Generated Audio 1', generated_audio_interp[0], sample_rate = 16000,global_step=epoch)
            writer.add_audio(f'{dataloader_type} Interpolation {batch_number}/Generated Audio 2', generated_audio_interp[-1], sample_rate = 16000,global_step=epoch)
            writer.add_audio(f'{dataloader_type} Interpolation {batch_number}/Audio interpolation', audio_cat, sample_rate = 16000,global_step=epoch)
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