# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from src.utils import *
# import torch.distributions as distrib
# import torchvision
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from cml.plot import cml_figure_matplotlib as figure


class AE(nn.Module):
    def __init__(self, in_channels=256):
        super(AE, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) #To find the device
        self.encoder = nn.Sequential(
            LayerEncoder(in_channels, in_channels//2, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(in_channels//2, in_channels//4, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(in_channels//4, in_channels//8, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            nn.Dropout1d(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            LayerDecoder(in_channels//8, in_channels//4, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=0),
            LayerDecoder(in_channels//4, in_channels//2, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=0),
            LayerDecoder(in_channels//2, in_channels, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=0),
            #nn.Dropout1d(p=0.2)
            Normalize()
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
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class LayerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv, output_padding_conv):
        super(LayerDecoder, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv, output_padding_conv)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class VAE(AE):
    
    def __init__(self, in_channels=256, encoding_dims=32*5,latent_dims=50,beta = 1):
        super(VAE, self).__init__(in_channels=in_channels)
        self.beta = beta
        self.latent_dims = latent_dims
        self.mu = nn.Sequential(nn.Linear(encoding_dims,latent_dims),
                                nn.Sigmoid())
        self.sigma = nn.Sequential(nn.Linear(encoding_dims,latent_dims),
                        nn.Softplus())
        self.decode_layer = nn.Sequential(
            nn.Linear(latent_dims,encoding_dims),
            nn.Tanh())
        # self.mu = nn.Sequential(
        #     nn.Conv1d(in_channels*4, in_channels*4, kernel_size=3,stride=2,padding=1),
        #     nn.ReLU(inplace=True),
        #     )
        # self.sigma = nn.Sequential(
        #     nn.Conv1d(in_channels*4, in_channels*4, kernel_size=3,stride=2,padding=1),
        #     nn.Softplus()
        #     )
        # self.decode_layer = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels*4, in_channels*4, kernel_size=3,stride=2,padding=1,output_padding=0),
        #     nn.ReLU(inplace=True)
        #     )

    def encode(self, x):
        x_hidden = self.encoder(x).view(-1,32*5)
        mu = self.mu(x_hidden)
        sigma = self.sigma(x_hidden)
        
        return mu, sigma
    
    def decode(self, z):
        z = self.decode_layer(z)
        z = z.view(-1,32,5)
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        mu, sigma = self.encode(x)
        # Obtain latent samples
        z, kl_div = self.latent(mu, sigma)
        # Decode the samples
        x_tilde = self.decode(z)
        return x_tilde, kl_div
    
    def latent(self, mu, sigma):
        device = self.dummy_param.device
        z = mu + sigma*torch.randn_like(sigma,device=device)
        kl_div = 0.5*(1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        return z, kl_div

    def compute_loss(self, x):
        recons_criterion = torch.nn.MSELoss(reduction='sum')
        x_tilde, kl = self.forward(x)
        full_loss = recons_criterion(x_tilde,x) - self.beta * kl
        
        return full_loss

def train_VAE(model, dataloader, epochs=5, lr=1e-3, device = torch.device("cpu"),writer=None):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # criterion_1 = torch.nn.MSELoss(reduction='sum')
    # criterion_2 = torch.nn.KLDivLoss(reduction='sum')
    loss_tensor = torch.tensor([]).to(device)
    
    for epoch in range(1, epochs + 1):
        full_loss = torch.Tensor([0]).to(device)
        for _, item in enumerate(dataloader):
            x = item['x']
            loss = model.compute_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            full_loss += loss
        loss_tensor = torch.cat([loss_tensor, full_loss])
        if writer is not None: 
            writer.add_scalar("Loss/train", full_loss.item(), epoch) 
        print('Step ',epoch,' over ',epochs,full_loss[0])
        
    return model, loss_tensor
