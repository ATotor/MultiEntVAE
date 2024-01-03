# -*- coding: utf-8 -*-
import os
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
    def __init__(self, in_channels=256):
        super(AE, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) #To find the device
        self.encoder = nn.Sequential(
            LayerEncoder(in_channels, in_channels*2, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(in_channels*2, in_channels*4, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            nn.Dropout1d(p=0.2)
        )
        
        self.decoder = nn.Sequential(
            LayerDecoder(in_channels*4, in_channels*2, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=0),
            LayerDecoder(in_channels*2, in_channels, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=0),
            nn.Dropout1d(p=0.2)
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
            nn.Conv1d(in_channels*4, in_channels*4, kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.sigma = nn.Sequential(
            nn.Conv1d(in_channels*4, in_channels*4, kernel_size=3,stride=2,padding=1),
            nn.Softplus()
            )
        self.decode_layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels*4, in_channels*4, kernel_size=3,stride=2,padding=1,output_padding=0),
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
        device = self.dummy_param.device
        z = mu + sigma*torch.randn_like(sigma,device=device)
        return z


def train_VAE(model, dataloader, epochs=5, lr=1e-3, device = torch.device("cpu"),writer=None):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion_1 = torch.nn.MSELoss(reduction='sum')
    criterion_2 = torch.nn.KLDivLoss(reduction='sum')
    loss_tensor = torch.tensor([]).to(device)
    
    for epoch in range(1, epochs + 1):
        full_loss = torch.Tensor([0]).to(device)
        for i, item in enumerate(dataloader):
            x = item['x']
            x_tilde = model(x)
            loss = criterion_1(x_tilde, x) + criterion_2(x_tilde, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            full_loss += loss
        loss_tensor = torch.cat([loss_tensor, full_loss])
        if writer is not None: 
            writer.add_scalar("Loss/train", full_loss.item(), epoch) 
        print('Step ',epoch,' over ',epochs,full_loss[0])
        
    return model, loss_tensor

def find_most_recent_VAE():
    found_files = []
    for _, _, files in os.walk("results"):
        for file in files:
            if len(file)>4 and file[:4] == "VAE_":
                strdate = file[4:]
                date_object = datetime.strptime(strdate,"%d-%m-%Y_%H_%M")
                found_files.append(date_object)
    if found_files:
        found_date = (sorted(found_files)[-1]).strftime("%d-%m-%Y_%H_%M")
        most_recent_file = "VAE_"+found_date
        return most_recent_file
    else:
        raise Exception("No model found")

def load_model(file_name):
    return torch.load('results/'+file_name)
    

def save_model(model):
    with torch.no_grad():
        date_time = datetime.now().strftime("%d-%m-%Y_%H_%M")
        torch.save(model, 
                   os.path.join("results","VAE_"+date_time))
def save_loss(loss):
    date_time = datetime.now().strftime("%d-%m-%Y_%H_%M")
    np.save(os.path.join("results","loss_"+date_time), loss)