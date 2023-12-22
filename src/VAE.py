# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
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
            LayerEncoder(in_channels, in_channels*2, 3, 2, 1),
            LayerEncoder(in_channels*2, in_channels*4, 3, 2, 1),
            nn.Dropout2d(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            LayerDecoder(in_channels*4, in_channels*2, 3, 2, 1, 1),
            LayerDecoder(in_channels*2, in_channels, 3, 2, 1, 1),
            nn.Dropout2d(p=0.2)
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
    
class LayerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv):
        super(LayerEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

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
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv, output_padding_conv)
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
            nn.Linear(4*7*7, latent_dims), 
            nn.ReLU()
        )
        self.sigma = nn.Sequential(
            nn.Linear(4*7*7, latent_dims),
            nn.Softplus())
        self.img_from_sample = nn.Sequential(
            nn.Linear(latent_dims, 4*7*7),
            nn.ReLU())
        
    def encode(self, x):
        
        x_hidden = self.encoder(x)
        x_hidden = torch.reshape(x_hidden, (-1, 4*7*7))
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
        z_tilde = torch.reshape(z_tilde, (-1, 4, 7, 7))
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
    loss_tensor = torch.tensor([])
    
    for epoch in range(1, epochs + 1):
        full_loss = torch.Tensor([0])
        for i, (x, _) in enumerate(dataloader):
            loss = criterion(model(x)[0], x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            full_loss += loss
        loss_tensor = torch.cat([loss_tensor, full_loss])
        print(full_loss)
        
    return model, loss_tensor


def save_model_and_loss(model, loss):
    with torch.no_grad():
        date_time = datetime.now().strftime("%d-%m-%Y_%H_%M")
        torch.save(model, 
                   './results/VAE_'+date_time)
        np.save('./results/loss_'+date_time, loss)
        
def disp_loss(loss):
    with torch.no_grad():
        plt.figure(figsize=(15,5))
        plt.plot(loss)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()