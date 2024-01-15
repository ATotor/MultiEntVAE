# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from src.utils import *
from tqdm import tqdm
from src.plots import *
# import torch.distributions as distrib
# import torchvision
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from cml.plot import cml_figure_matplotlib as figure


class AE(nn.Module):
    def __init__(self, in_channels,hidden_dims):
        super(AE, self).__init__()
        self.hidden_dim = hidden_dims[-1]
        self.encoder = nn.Sequential(
            LayerEncoder(in_channels, hidden_dims[0], kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(hidden_dims[0], hidden_dims[1], kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(hidden_dims[1], hidden_dims[2], kernel_size_conv=3,stride_conv=2,padding_conv=1),
        )
        
        self.decoder = nn.Sequential(
            LayerDecoder(hidden_dims[2], hidden_dims[1], kernel_size_conv=3),
            LayerDecoder(hidden_dims[1], hidden_dims[0], kernel_size_conv=3),
            LayerDecoder(hidden_dims[0], in_channels, kernel_size_conv=3),
            nn.Conv1d(in_channels,in_channels,kernel_size=3,padding='same'),
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class LayerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv):
        super(LayerEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class LayerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv):
        super(LayerDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size_conv, padding = "same")
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        #x = self.batchnorm(x)
        x = self.relu(x)

        return x


class VAE(AE):
    
    def __init__(self, in_channels, latent_dims,hidden_dims,beta):
        super(VAE, self).__init__(in_channels=in_channels,hidden_dims=hidden_dims)
        self.recons_criterion = torch.nn.MSELoss(reduction='sum')
        self.encoding_shape_time = 16
        self.beta = beta
        self.latent_dims = latent_dims

        self.mu=nn.Conv1d(in_channels=self.hidden_dim,out_channels=latent_dims,kernel_size=3,padding="same")
        self.logvar=nn.Conv1d(in_channels=self.hidden_dim,out_channels=latent_dims,kernel_size=3,padding="same")

        self.decode_layer = nn.Conv1d(in_channels=latent_dims,out_channels=self.hidden_dim,kernel_size=3,padding="same")


    def encode(self, x):
        
        x_hidden = self.encoder(x)
        mu = self.mu(x_hidden)
        logvar = self.logvar(x_hidden)
        
        return mu, logvar
    
    def decode(self, z):
        z = self.decode_layer(z)
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        mu, logvar = self.encode(x)
        # Obtain latent samples
        z, kl_div = self.latent(mu, logvar)
        # Decode the samples
        x_tilde = self.decode(z)
        return x_tilde, kl_div
    
    def latent(self, mu, logvar):
        std=torch.exp(0.5*logvar)
        z = mu + std*torch.randn_like(std,device=std.device)
        kl_div = 0.5*(-1 - logvar + mu**2 + logvar.exp()).sum()
        return z, kl_div

    def compute_loss(self, x):
        x_tilde, kl = self.forward(x)
        mse_loss = self.recons_criterion(x_tilde,x)
        full_loss = mse_loss + self.beta * kl
        return full_loss, mse_loss, kl
    
    def generate(self,device):
        x = torch.randn((8,self.latent_dims,self.encoding_shape_time),device=device)
        return self.decode(x)
        

def train_VAE(model:nn.Module, 
              dataloader:torch.utils.data.Dataset, 
              valid_dummy_item1:dict, 
              train_dummy_item1:dict,
              valid_dummy_item2:dict, 
              train_dummy_item2:dict,
              training_norm: nn.Module,
              training_denorm: nn.Module,
              valid_norm:nn.Module,
              valid_denorm:nn.Module,
              inverse_transform:nn.Module,
              writer:torch.utils.tensorboard.SummaryWriter, 
              epochs:int=5, 
              lr:float=1e-3, 
              device:torch.device="cpu",
              saving_model_file:str="",
              ):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # criterion_1 = torch.nn.MSELoss(reduction='sum')
    # criterion_2 = torch.nn.KLDivLoss(reduction='sum')
    global_step=0
    for epoch in tqdm(range(1, epochs + 1)):
        full_loss = torch.Tensor([0]).to(device)
        full_mse = torch.Tensor([0]).to(device)
        full_kl = torch.Tensor([0]).to(device)
        for i, item in tqdm(enumerate(dataloader),total=len(dataloader),desc=f"Epoch {epoch} over {epochs}",leave=False):
            x = item['x']
            batch_size = x.shape[0]
            x = training_norm(x)
            loss, mse_loss, kl_div = model.compute_loss(x)
            full_loss += loss
            full_mse += mse_loss
            full_kl += kl_div
            loss /= batch_size
            optimizer.zero_grad()
            #log_model_grad_norm(model,writer,global_step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            global_step+=1

        if saving_model_file :
            torch.save(model,saving_model_file)
        #loss_tensor = torch.cat([loss_tensor, full_loss])
            
        #---------------------------Writing on logs-------------------------------------

        tqdm.write(f'Epoch {epoch}\tFull loss: {full_loss.item():0.2e}\tReconstruction loss: {full_mse.item():0.2e}\tKl divergence: {full_kl.item():0.2e}')
        log_model_loss(writer, full_loss, full_mse, full_kl, epoch)

        tensorboard_writer( model=model,
                            item1=train_dummy_item1, 
                            item2=train_dummy_item2,
                            dataloader_type = "Training",
                            writer=writer,
                            inverse_transform=nn.Sequential(training_denorm,inverse_transform),
                            normalizer=training_norm,
                            batch_size=3,
                            epoch=epoch,
                            )

        tensorboard_writer( model=model,
                            item1=valid_dummy_item1,
                            item2=valid_dummy_item2,
                            dataloader_type = "Validation",
                            writer=writer,
                            inverse_transform=nn.Sequential(valid_denorm,inverse_transform),
                            normalizer=valid_norm,
                            batch_size=3,
                            epoch=epoch,
                            )
    return model
