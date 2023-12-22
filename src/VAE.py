# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributions as distrib
# import torchvision
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from cml.plot import cml_figure_matplotlib as figure


class AE(nn.Module):
    def __init__(self, in_channels=1):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            Conv2DLayerEncoder(in_channels, in_channels*2, 2, 1, 0, 2, 1, 0),
            Conv2DLayerEncoder(in_channels*2, in_channels*4, 2, 1, 0, 2, 1, 0),
            Conv2DLayerEncoder(in_channels*4, in_channels*8, 2, 1, 0, 2, 1, 0),
            Conv2DLayerEncoder(in_channels*8, in_channels*16, 2, 1, 0, 2, 1, 0),
            nn.Dropout2d(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            Conv2DLayerDecoder(in_channels*16, in_channels*8, 2, 1, 0, 2),
            Conv2DLayerDecoder(in_channels*8, in_channels*4, 2, 1, 0, 2),
            Conv2DLayerDecoder(in_channels*4, in_channels*2, 2, 1, 0, 2),
            Conv2DLayerDecoder(in_channels*2, in_channels, 2, 1, 0, 2),
            nn.Dropout2d(p=0.2)
            )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.upsample(encoded, scale_factor=1.2, mode='bilinear')
        decoded = self.decoder(encoded)
        return decoded
    
    
    
class Conv2DLayerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv, kernel_size_pool, stride_pool, 
                 padding_pool):
        super(Conv2DLayerEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size_pool, stride_pool, padding_pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
    
class Conv2DLayerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv, up_scale):
        super(Conv2DLayerDecoder, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class VAE(AE):
    
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(6400, latent_dims), 
            nn.ReLU()
        )
        self.sigma = nn.Sequential(nn.Linear(6400, latent_dims),
                                   nn.Softplus())
        self.img_from_sample = nn.Sequential(nn.Linear(latent_dims, 6400),
                                            nn.ReLU())
        
    def encode(self, x):
        
        x_hidden = self.encoder(x)
        x_hidden = torch.reshape(x_hidden, (64, -1))
        mu = self.mu(x_hidden)
        sigma = self.sigma(x_hidden)
        
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(x, z_params)
        z_tilde = self.img_from_sample(z_tilde)
        z_tilde = torch.reshape(z_tilde, (64, 16, 20, 20))
        # Upsample
        z_tilde = F.upsample(z_tilde, scale_factor=1.2, mode='bilinear')
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, kl_div
    
    def latent(self, x, z_params):
        z = z_params[0] + torch.randn(z_params[0].shape[-1]) * z_params[1]
        kl_div = 1/2*torch.sum(1 + torch.log((z_params[1])**2) - 
                               (z_params[0])**2 - (z_params[1])**2)
        
        return z, kl_div


def train_VAE(model, dataloader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss(reduction='sum')
    
    for epoch in range(1, epochs + 1):
        full_loss = torch.Tensor([0])
        for i, (x, _) in enumerate(dataloader):
            if (x.shape[0]!=64): # CHECK DATALOADER SIZE
                pass
            loss = criterion(model(x)[0], x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            full_loss += loss
        print(full_loss)
        
    return model