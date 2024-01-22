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
    def __init__(self, in_channels=1):
        super(AE, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) #To find the device
        self.encoder = nn.Sequential(
            LayerEncoder(in_channels, 8, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(8, 16, kernel_size_conv=3,stride_conv=2,padding_conv=1),
            LayerEncoder(16, 32, kernel_size_conv=3,stride_conv=1,padding_conv=0),
            LayerEncoder(32, 64, kernel_size_conv=3,stride_conv=1,padding_conv=0),
        )
        self.decoder = nn.Sequential(
            LayerDecoder(64,32, kernel_size_conv=3,stride_conv=1,padding_conv=0,output_padding_conv=0),
            LayerDecoder(32,16, kernel_size_conv=3,stride_conv=1,padding_conv=0,output_padding_conv=0),
            LayerDecoder(16,8, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=1),
            LayerDecoder(8,in_channels, kernel_size_conv=3,stride_conv=2,padding_conv=1,output_padding_conv=1),
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
    def __init__(self, in_channels=1, latent_dims=32,beta= 1):
        super(VAE, self).__init__(in_channels)
        self.recons_criterion = torch.nn.MSELoss(reduction='sum')
        self.beta = beta
        self.latent_dims = latent_dims
        self.num_classes=10
        self.hidden_dim=3
        self.encoding_dims=256
        self.mu = nn.Sequential(nn.Linear(self.encoding_dims,self.latent_dims),nn.Tanh())
        self.sigma = nn.Sequential(nn.Linear(self.encoding_dims,self.latent_dims),nn.Softplus())
        self.flatten=nn.Flatten()
        self.channel_end=64
        self.linear_encode=nn.Sequential(nn.Linear(3*3*self.channel_end,self.encoding_dims),nn.ReLU())
        self.decode_layer = nn.Sequential(nn.Linear(self.latent_dims,3*3*self.channel_end),nn.ReLU())
        self.label_projector=nn.Sequential(nn.Conv1d(in_channels=self.num_classes, out_channels=self.latent_dims, kernel_size=3,padding='same'),nn.ReLU())
        
    def encode(self, x):
        x_hidden = self.encoder(x)
        x_hidden=self.flatten(x_hidden)
        x_latent=self.linear_encode(x_hidden)
        mu = self.mu(x_latent)
        sigma = self.sigma(x_latent)
        
        return mu, sigma
    def condition_on_label(self,z:torch.Tensor,label:torch.Tensor):
        y = nn.functional.one_hot(label.long(),num_classes=self.num_classes)
        y = y.reshape((-1,self.num_classes,1))
        projected_label = self.label_projector(y.float())
        projected_label=projected_label.squeeze()
        return z + projected_label
    
    def decode(self, z):
        z = self.decode_layer(z)
        z = z.view(-1,self.channel_end,self.hidden_dim,self.hidden_dim)
        return self.decoder(z)
    
    def forward(self, x,label=None):
        # Encode the inputs
        mu, sigma = self.encode(x)
        # Obtain latent samples
        z, kl_div = self.latent(mu, sigma)
        # Decode the samples
        z = z if label is None else self.condition_on_label(z, label)
        x_tilde = self.decode(z)
        return x_tilde, kl_div
    
    def latent(self, mu, sigma):
        device = self.dummy_param.device
        z = mu + sigma*torch.randn_like(sigma,device=device)
        kl_div = 0.5*(-1 - torch.log(sigma**2) + mu**2 + sigma**2).sum()
        return z, kl_div

    def compute_loss(self, x,label=None):
        x_tilde, kl = self.forward(x,label)
        mse_loss = self.recons_criterion(x_tilde,x)
        full_loss = mse_loss + self.beta * kl
        return full_loss, mse_loss, kl

def train_VAE(model, dataloader,test_dataloader, epochs=20, lr=1e-3, device = torch.device("cuda"),writer=None, spec_normalizer=lambda x:x,starting_time=""):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # criterion_1 = torch.nn.MSELoss(reduction='sum')
    # criterion_2 = torch.nn.KLDivLoss(reduction='sum')
    
    for epoch in range(1, epochs + 1):
        full_loss = torch.Tensor([0]).to(device)
        full_mse = torch.Tensor([0]).to(device)
        full_kl = torch.Tensor([0]).to(device)
        full_loss_val = torch.Tensor([0]).to(device)
        full_mse_val = torch.Tensor([0]).to(device)
        full_kl_val = torch.Tensor([0]).to(device)
        for i, item in tqdm(enumerate(dataloader),total=len(dataloader),desc=f"Epoch {epoch} over {epochs}",position=1,leave=False):
            tqdm._instances.clear()
            model.train()
            x = item[0]
            label=item[1]
            batch_size = x.shape[0]
            x = spec_normalizer(x)
            loss, mse_loss, kl_div = model.compute_loss(x,label)
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
            for i,item in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc=f"val_epoch {epoch} over  {epochs}",position=2,leave=False):
               x = item[0]
               label=item[1]
               batch_size = x.shape[0]
               x = spec_normalizer(x)
               loss_val, mse_loss_val, kl_div_val = model.compute_loss(x,label)
               full_loss_val += loss_val
               full_mse_val += mse_loss_val
               full_kl_val += kl_div_val
        fig=disp_MNIST_example(model, test_dataloader)
        writer.add_figure("Image/input,recons",fig,epoch)
        if starting_time :
            torch.save(model,starting_time)
        #loss_tensor = torch.cat([loss_tensor, full_loss])
        if writer is not None: 
            tqdm.write(f' Epoch {epoch}\tFull loss vall: {full_loss_val.item():0.2e}\tReconstruction loss val: {full_mse_val.item():0.2e}\tKl divergence val: {full_kl_val.item():0.2e}')
            log_model_loss(writer, full_loss,full_loss_val, full_mse,full_mse_val, full_kl,full_kl_val, epoch)
            fig=disp_MNIST_example(model, test_dataloader)
            writer.add_figure("Image/input,recons",fig,epoch)
            #log_model_grad_norm(model,writer,epoch)
            
    return model
beta=1
batchsize=64
latent_dims=16
model_vae=VAE(in_channels=1, latent_dims=latent_dims,beta = beta)
dataloader,test_dataloader=MNIST_give_dataloader(root='.', batch_size=batchsize)
path_dir="C:/Users/alexa/OneDrive/Documents/Code en tout genre/Python Scripts/script_auto_enco/"
writer = SummaryWriter(path_dir+f"/runs3/model_CVAE_beta{beta}_opti_archi_lattent_{latent_dims}")
dataiter = iter(dataloader)
image,label = next(dataiter)
model_vae.eval()
writer.add_graph(model_vae,image)
model_vae.train()
model=train_VAE(model_vae, dataloader,test_dataloader,writer=writer)
Y1=np.random.randint(0,9)
fig=disp_Mnist_random_sample_condition(model,H,W,Y1)
writer.add_figure(f"Image/CVAE_Recons_conditional_beta={beta},label={Y1}_lattent_{latent_dims}",fig)
summary(model, input_size=(batchsize, 1, 28, 28))
torch.save(model.state_dict(),path_dir+f"/model_save/model_CVAE_beta{beta}_opti_archi_lattent_{latent_dims}")
