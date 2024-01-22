# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:27:46 2024

@author: alexa
"""
import os
import torch
import torch.nn as nn
from utils import *
from tqdm import tqdm
from plots import *
from dataset_loader import MNIST_give_dataloader
from torchinfo import summary
import json
import pandas as pd

def cal_depht(N,num_layers): #return the correct channels number given a seed
    in_channels=[]
    out_channels=[]
    for j in range(num_layers):
        if j ==0:
            in_channels.append(1)
        else:
            in_channels.append(N*np.power(2,j-1))
        out_channels.append(N*np.power(2,j))
    return in_channels,out_channels
#The implementation of AE, VAE is quite the same exept it's modulable to allow the
#grid search to process smoothly.  

class AE_mod(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size_conv_in,stride_conv_in, padding_conv_in,kernel_size_conv_out, 
    stride_conv_out, padding_conv_out, output_padding_conv,num_layers=4,func_acti=nn.ReLU()):
        super(AE_mod, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) #To find the device
        self.encoder = LayerEncoder_mod(in_channels,out_channels, kernel_size_conv_in, 
        stride_conv_in, padding_conv_in,num_layers)
        self.decoder = LayerDecoder_mod(list(reversed(out_channels)),list(reversed(in_channels)), kernel_size_conv_out, 
        stride_conv_out, padding_conv_out, output_padding_conv,num_layers)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LayerEncoder_mod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, stride_conv, padding_conv,num_layers,func_acti=nn.ReLU()):
        super(LayerEncoder_mod, self).__init__()
        self.conv_lay=nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels[i], out_channels[i], kernel_size_conv[i], 
                              stride_conv[i], padding_conv[i]),func_acti) for i in range(num_layers)])
    def forward(self, x):
        for i, l in enumerate(self.conv_lay):
            x = self.conv_lay[i](x)
        return x

class LayerDecoder_mod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, 
                 stride_conv, padding_conv, output_padding_conv,num_layers,func_acti=nn.ReLU()):
        super(LayerDecoder_mod, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.deconv_lay =nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(in_channels[i], out_channels[i], kernel_size_conv[i], 
                              stride_conv[i], padding_conv[i], output_padding_conv[i]),func_acti) for i in range(num_layers)])
        self.conv_end=nn.Conv2d(out_channels[-1],out_channels[-1],kernel_size=3,padding='same')
  
    def forward(self, x):
        for i, l in enumerate(self.deconv_lay):
            x = self.deconv_lay[i](x)
        x=self.conv_end(x)
        return x

class VAE_mod(AE_mod):
    def __init__(self,in_channels,out_channels, kernel_size_conv_in,stride_conv_in, padding_conv_in,kernel_size_conv_out, 
    stride_conv_out, padding_conv_out, output_padding_conv,latent_dim,beta,num_layers=4,func_acti=nn.ReLU()):
        super(VAE_mod, self).__init__(in_channels,out_channels, kernel_size_conv_in,stride_conv_in, padding_conv_in,kernel_size_conv_out, 
        stride_conv_out, padding_conv_out, output_padding_conv,num_layers=4,func_acti=nn.ReLU())
        self.recons_criterion = torch.nn.MSELoss(reduction='sum')
        self.beta = beta
        self.latent_dims = latent_dim
        self.hidden_dim=3
        self.encoding_dims=256
        self.mu = nn.Sequential(nn.Linear(self.encoding_dims,latent_dim),nn.Tanh())
        self.sigma = nn.Sequential(nn.Linear(self.encoding_dims,latent_dim),nn.Softplus())
        self.flatten=nn.Flatten()
        self.channel_end=out_channels[-1]
        self.linear_encode=nn.Sequential(nn.Linear(3*3*self.channel_end,self.encoding_dims),nn.ReLU())
        self.decode_layer = nn.Sequential(nn.Linear(latent_dim,3*3*self.channel_end),nn.ReLU())

    def encode(self, x):
        x_hidden = self.encoder(x)
        x_hidden=self.flatten(x_hidden)
        x_latent=self.linear_encode(x_hidden)
        mu = self.mu(x_latent)
        sigma = self.sigma(x_latent)
        
        return mu, sigma
    
    def decode(self, z):
        z = self.decode_layer(z)
        z = z.view(-1,self.channel_end,self.hidden_dim,self.hidden_dim)
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

device=torch.device("cuda")

def train_VAE_grid_search(model,train_dataloader,test_dataloader,lr=0.001,epochs=6,device=device,writer=None,spec_normalizer=lambda x:x,starting_time=""):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for epoch in tqdm(range(1, epochs + 1),total=epochs,leave=False,position=4,desc="through epochs"):
        tqdm._instances.clear()
        full_loss = torch.Tensor([0]).to(device)
        full_mse = torch.Tensor([0]).to(device)
        full_kl = torch.Tensor([0]).to(device)
        full_loss_val = torch.Tensor([0]).to(device)
        full_mse_val = torch.Tensor([0]).to(device)
        full_kl_val = torch.Tensor([0]).to(device)
        for i, item in tqdm(enumerate(train_dataloader),total=len(train_dataloader),desc=f"through the train_datas",leave=False,position=5):
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
            for i,item in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc="through the test_datas",leave=False,position=6):
               x = item[0]
               batch_size = x.shape[0]
               x = spec_normalizer(x)
               loss_val, mse_loss_val, kl_div_val = model.compute_loss(x)
               full_loss_val += loss_val
               full_mse_val += mse_loss_val
               full_kl_val += kl_div_val
        #loss_tensor = torch.cat([loss_tensor, full_loss])
        if writer is not None: 
            tqdm.write(f'Epoch {epoch}\tFull loss: {full_loss.item():0.2e}\tFull loss vall: {full_loss_val.item():0.2e}\tReconstruction loss: {full_mse.item():0.2e}\tReconstruction loss val: {full_mse_val.item():0.2e}\tKl divergence: {full_kl.item():0.2e}\tKl divergence val: {full_kl_val.item():0.2e}')
            log_model_loss(writer, full_loss,full_loss_val, full_mse,full_mse_val, full_kl,full_kl_val, epoch)
            fig=disp_MNIST_example(model, test_dataloader)
            writer.add_figure("Image/input,recons",fig,epoch)
            fig=disp_MNIST_example(model, test_dataloader)
            writer.add_figure("Image/input,recons",fig,epoch)
            #log_model_grad_norm(model,writer,epoch)
            
    return model,[full_loss.item(),full_mse.item(),full_kl.item(),full_loss_val.item(),full_mse_val.item(),full_kl_val.item()]


kernel_size_conv_in=[3,3,3,3]
stride_conv_in=[2,2,1,1]
padding_conv_in=[1,1,0,0]

kernel_size_conv_out=[3,3,3,3]
stride_conv_out=[1,1,2,2]
padding_conv_out=[0,0,1,1]
output_padding_conv=[0,0,1,1]

seed_dephts=[4,8,16,32]
betas=[0.1,0.5,1,2]
lattent_dims=[4,8,16,32]
activa_func=["silu","relu"]
path_dir="C:/Users/alexa/OneDrive/Documents/Code en tout genre/Python Scripts/script_auto_enco/"



precedent_result=[]
path="C:/Users/alexa/OneDrive/Documents/Code en tout genre/Python Scripts/script_auto_enco/result_grid_search"
with open(path+"/VAE_grid_search_result.txt",'r') as fp:
    for line in fp:
        try:
            precedent_result.append(json.loads(line))
        except :
            pass           
df_precedent_result = pd.DataFrame(precedent_result)
already_happen_combination=[]
df_result=df_precedent_result
for i in range(len(df_result)):
  already_happen_combination.append((int(df_result.iloc[i]['seed_dephts']),
                                   float(df_result.iloc[i]['betas']),
                                   int(df_result.iloc[i]['lattent_dims']),
                                   str(df_result.iloc[i]['activation_function'])))
device=torch.device("cuda")
train_dataloader,test_dataloader=MNIST_give_dataloader(root='.', batch_size=64)
for S in tqdm(seed_dephts,total=len(seed_dephts),position=0,leave=False,desc="through different depht"):
    for beta in tqdm(betas,total=len(betas),position=1,leave=False,desc=f"through different betas"):
        for lattent_dim in tqdm(lattent_dims,total=len(lattent_dims),position=2,leave=False,desc=f"through different lattent_dim"):
            for acti_fun in tqdm(activa_func,total=len(activa_func),position=3,leave=False,desc=f"through different activation function"):
                device=torch.device("cuda")
                #writer = SummaryWriter(path_dir+f"/runs2/vae_beta={beta}_lattent_dim={lattent_dim}_acti_fun={acti}_seed_dephts={S}")
                if((S,beta,lattent_dim,acti_fun) in already_happen_combination):
                   continue
                in_channels,out_channels=cal_depht(S,4)
                if acti_fun=="relu":
                    model=VAE_mod(in_channels,out_channels,kernel_size_conv_in,stride_conv_in,padding_conv_in
                              ,kernel_size_conv_out,stride_conv_out,padding_conv_out,output_padding_conv,
                              lattent_dim,beta,4,nn.ReLU())
                elif acti_fun=="silu":
                    model=VAE_mod(in_channels,out_channels,kernel_size_conv_in,stride_conv_in,padding_conv_in
                              ,kernel_size_conv_out,stride_conv_out,padding_conv_out,output_padding_conv,
                              lattent_dim,beta,4,nn.SiLU())
                nb_paramètre= sum(p.numel() for p in model.parameters())
                model_fin,loss_all_end=train_VAE_grid_search(model,train_dataloader,test_dataloader,lr=0.001,epochs=6,device=device,writer=None)
                result={'seed_dephts':S,'betas':beta,'lattent_dims':lattent_dim,'activation_function':acti_fun,
                        'loss_kl_val':loss_all_end[5],'loss_MSE_val':loss_all_end[4],'loss_tot_val':loss_all_end[3],
                        'nb_parameter':nb_paramètre}
                with open(path+"/VAE_grid_search_result.txt","a") as file:
                   result = json.dumps(result)
                   file.write(result)
                   file.write('\n')
df_result.to_csv("resultat_grid_search_vae.csv")
    
