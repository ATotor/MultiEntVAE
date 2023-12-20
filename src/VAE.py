# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
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
            Conv2DLayerEncoder(in_channels, in_channels*2, 2, 0, 0, 2, 0, 0),
            Conv2DLayerEncoder(in_channels*2, in_channels*4, 2, 1, 0, 2, 1, 0),
            Conv2DLayerEncoder(in_channels*4, in_channels*8, 2, 0, 0, 2, 0, 0),
            Conv2DLayerEncoder(in_channels*8, in_channels*16, 2, 1, 0, 2, 1, 0),
            Conv2DLayerEncoder(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            Conv2DLayerDecoder(in_channels*16, in_channels*8, 2, 1, 0, 2),
            Conv2DLayerDecoder(in_channels*8, in_channels*4, 2, 0, 0, 2),
            Conv2DLayerDecoder(in_channels*4, in_channels*2, 2, 1, 0, 2),
            Conv2DLayerDecoder(in_channels*2, in_channels, 2, 0, 0, 2),
            nn.Dropout2d(p=0.2)
            )

    def forward(self, x):
        encoded = self.encoder(x)
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
        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class VAE(AE):
    
    def __init__(self):
        super(VAE, self).__init__(latent_dims)
        self.mu = nn.Sequential(
            nn.Linear(1, 
                      latent_dims), 
            nn.ReLU()
        )
        self.sigma = nn.Sequential(nn.Linear(self.encoding_dims, 
                                             self.latent_dims),
                                   nn.Softplus())
        
    def encode(self, x):
        
        x_hidden = self.encoder(x)
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
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde.reshape(-1, 1, 28, 28), kl_div
    
    def latent(self, x, z_params):
        z = z_params[0] + torch.randn(z_params[0].shape[-1]) * z_params[1]
        kl_div = 1/2*torch.sum(1 + torch.log((z_params[1])**2) - (z_params[0])**2 - (z_params[1])**2)
        
        return z, kl_div
    
