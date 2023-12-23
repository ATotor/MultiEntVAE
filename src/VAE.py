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
            LayerEncoder(in_channels, in_channels*6, 4, 2, 3),
            LayerEncoder(in_channels*6, in_channels*12, 3, 2, 1),
            nn.Dropout2d(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            LayerDecoder(in_channels*12, in_channels*6, 3, 2, 1, 1),
            LayerDecoder(in_channels*6, in_channels, 3, 2, 3, 1),
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
    
    def __init__(self, in_channels=1):
        super(VAE, self).__init__()
        self.mu = nn.Sequential(
            nn.Conv2d(in_channels*12, in_channels*12, 3, 2, 1),
            nn.ReLU(inplace=True),
            )
        self.sigma = nn.Sequential(
            nn.Conv2d(in_channels*12, in_channels*12, 3, 2, 1),
            nn.Softplus()
            )
        self.decode_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels*12, in_channels*12, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
            )
        
    def encode(self, x):
        
        x_hidden = self.encoder(x)
        mu = self.mu(x_hidden)
        sigma = self.sigma(x_hidden)
        
        return mu, sigma
    
    def decode(self, z):
        z = self.decode_layer(z)
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        mu, sigma = self.encode(x)
        # Obtain latent samples
        z = self.latent(x, mu, sigma)
        # Decode the samples
        x_tilde = self.decode(z)
        return x_tilde
    
    def latent(self, x, mu, sigma):
        z = mu + torch.randn(sigma.shape[-3], sigma.shape[-2], sigma.shape[-1]) * sigma
        return z


def train_VAE(model, dataloader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion_1 = torch.nn.MSELoss(reduction='sum')
    criterion_2 = torch.nn.KLDivLoss(reduction='sum')
    loss_tensor = torch.tensor([])
    
    for epoch in range(1, epochs + 1):
        full_loss = torch.Tensor([0])
        for i, (x, _) in enumerate(dataloader):
            x_tilde = model(x)
            #print(criterion_1(x_tilde, x), criterion_2(x_tilde, x))
            loss = criterion_1(x_tilde, x) + criterion_2(x_tilde, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            full_loss += loss
        loss_tensor = torch.cat([loss_tensor, full_loss])
        print('Step ',epoch,' over ',epochs,full_loss[0])
        
    return model, loss_tensor


def load_model(file_name):
    return torch.load('results/'+file_name)
    

def save_model_and_loss(model, loss):
    with torch.no_grad():
        date_time = datetime.now().strftime("%d-%m-%Y_%H_%M")
        torch.save(model, 
                   './results/VAE_'+date_time)
        np.save('./results/loss_'+date_time, loss)