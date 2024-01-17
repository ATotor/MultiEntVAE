# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from utils import *
from tqdm.auto import tqdm
from plots import *
from dataset_loader import MNIST_give_dataloader
from torchinfo import summary
# import torch.distributions as distrib
# import torchvision
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from cml.plot import cml_figure_matplotlib as figure


class AE(nn.Module):
    def __init__(self, in_channels=28,hidden_dim=10):
        super(AE, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) #To find the device
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            LayerEncoder(in_channels, hidden_dim, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(hidden_dim, hidden_dim, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(hidden_dim, hidden_dim, kernel_size_conv=7,stride_conv=1,padding_conv=0),
        )
        self.decoder = nn.Sequential(
            LayerDecoder(hidden_dim, hidden_dim, kernel_size_conv=7,stride_conv=1,padding_conv=0,output_padding_conv=0),
            LayerDecoder(hidden_dim, hidden_dim, kernel_size_conv=4,stride_conv=2,padding_conv=1,output_padding_conv=0),
            LayerDecoder(hidden_dim, in_channels, kernel_size_conv=4,stride_conv=2,padding_conv=1,output_padding_conv=0),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding='same'),
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        #x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class LayerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv, output_padding_conv):
        super(LayerDecoder, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size_conv, 
                              stride_conv, padding_conv, output_padding_conv)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = self.upsample(x)
        x = self.conv(x)
        #x = self.batchnorm(x)
        x = self.relu(x)

        return x


class VAE(AE):
    def __init__(self, in_channels=1, latent_dims=50,hidden_dim=10,beta = 1):
        super(VAE, self).__init__(in_channels=in_channels,hidden_dim=hidden_dim)
        self.recons_criterion = torch.nn.MSELoss(reduction='sum')
        self.encoding_shape2 = 1
        encoding_dims = hidden_dim * self.encoding_shape2
        self.beta = beta
        self.latent_dims = latent_dims
        self.linear_encode=nn.Sequential(nn.Linear(hidden_dim ,encoding_dims),
                                nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(encoding_dims,latent_dims),
                                nn.Tanh())
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
        x_hidden = self.encoder(x)
        x_hidden=x_hidden.view(-1,self.hidden_dim*self.encoding_shape2)
        mu = self.mu(x_hidden)
        sigma = self.sigma(x_hidden)
        
        return mu, sigma
    
    def decode(self, z):
        z = self.decode_layer(z)
        z = z.view(-1,self.hidden_dim,self.encoding_shape2,self.encoding_shape2)
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
        kl_div = 0.5*(-1 - torch.log(sigma**2) + mu**2 + sigma**2).sum()
        return z, kl_div

    def compute_loss(self, x):
        x_tilde, kl = self.forward(x)
        mse_loss = self.recons_criterion(x_tilde,x)
        full_loss = mse_loss + self.beta * kl
        return full_loss, mse_loss, kl

def train_VAE(model, dataloader,test_dataloader, epochs=20, lr=1e-3, device = torch.device("cuda"),writer=None, spec_normalizer=lambda x:x,starting_time=""):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # criterion_1 = torch.nn.MSELoss(reduction='sum')
    # criterion_2 = torch.nn.KLDivLoss(reduction='sum')
    
    for epoch in range(1, epochs + 1):
        tqdm._instances.clear()
        full_loss = torch.Tensor([0]).to(device)
        full_mse = torch.Tensor([0]).to(device)
        full_kl = torch.Tensor([0]).to(device)
        full_loss_val = torch.Tensor([0]).to(device)
        full_mse_val = torch.Tensor([0]).to(device)
        full_kl_val = torch.Tensor([0]).to(device)
        for i, item in tqdm(enumerate(dataloader),total=len(dataloader),desc=f"Epoch {epoch} over {epochs}",position=1,leave=True):
            tqdm._instances.clear()
            model.train()
            x = item[0]
            batch_size = x.shape[0]
            x = spec_normalizer(x)
            loss, mse_loss, kl_div = model.compute_loss(x)
            full_loss += loss
            full_mse += mse_loss
            full_kl += kl_div
            loss /= batch_size
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        with torch.no_grad():
            model.eval()
            for i,item in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc=f"val_epoch {epoch} over  {epochs}",position=2,leave=True):
               tqdm._instances.clear()
               x = item[0]
               batch_size = x.shape[0]
               x = spec_normalizer(x)
               loss_val, mse_loss_val, kl_div_val = model.compute_loss(x)
               full_loss_val += loss_val
               full_mse_val += mse_loss_val
               full_kl_val += kl_div_val
        fig=disp_MNIST_example(model, test_dataloader)
        writer.add_figure("Image/input,recons",fig,epoch)
        if starting_time :
            torch.save(model,starting_time)
        #loss_tensor = torch.cat([loss_tensor, full_loss])
        if writer is not None: 
            tqdm.write(f'Epoch {epoch}\tFull loss: {full_loss.item():0.2e}\tFull loss vall: {full_loss_val.item():0.2e}\tReconstruction loss: {full_mse.item():0.2e}\tReconstruction loss val: {full_mse_val.item():0.2e}\tKl divergence: {full_kl.item():0.2e}\tKl divergence val: {full_kl_val.item():0.2e}')
            log_model_loss(writer, full_loss,full_loss_val, full_mse,full_mse_val, full_kl,full_kl_val, epoch)
            fig=disp_MNIST_example(model, test_dataloader)
            writer.add_figure("Image/input,recons",fig,epoch)
            #log_model_grad_norm(model,writer,epoch)
            
    return model
beta=2
batchsize=64
model_vae=VAE(in_channels=1, latent_dims=100,hidden_dim=100,beta = beta)
dataloader,test_dataloader=MNIST_give_dataloader(root='.', batch_size=batchsize)
path_dir="C:/Users/alexa/OneDrive/Documents/Code en tout genre/Python Scripts/script_auto_enco/"
writer = SummaryWriter(path_dir+f"/runs/vae_beta={beta}_lattent_hid=100")
model=train_VAE(model_vae, dataloader,test_dataloader,writer=writer)
summary(model, input_size=(batchsize, 1, 28, 28))
torch.save(model.state_dict(),path_dir+f"/model_save/model_beta{beta}_lattent_100")
