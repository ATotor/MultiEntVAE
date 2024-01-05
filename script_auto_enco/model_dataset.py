# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 19:53:03 2023

@author: alexa
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Spectrogram,GriffinLim
import pandas as pd

class NsynthDataset(Dataset):
    def __init__(self,json_file,sound_dir,transform = Spectrogram(n_fft=1023,hop_length=250,power=2,normalized=True)):
        super().__init__()
        self.json = pd.read_json(json_file,orient="index")
        self.json=self.json.loc[self.json["instrument_source_str"]=="acoustic"]
        self.json=self.json.loc[self.json["pitch"]>=36]# Do 2
        self.json=self.json.loc[self.json["pitch"]<=96]# Do 7
        self.sound_dir=sound_dir
        self.transform = lambda x : transform(x)#[0] En fonction de s'il y a 1 channel ou nfft/2 channels

    def __len__(self):
        return len(self.json)
    
    def __getitem__(self, idx):
        note_str = self.json['note_str'].iloc[idx]
        waveform, sampling_rate = torchaudio.load(self.sound_dir+"/"+note_str+".wav")
        waveform = waveform[:, :32000]
        env = torch.ones_like(waveform)
        env[:,-8000:] = torch.linspace(1, 0, 8000)
        waveform = waveform * env

        spectrogram = self.transform(waveform)
        return spectrogram, self.json['pitch'].iloc[idx]


class Encodeur(nn.Module):
    def __init__(self,lattent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,lattent_dim,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),   
        )

    def forward(self, x):
        output = self.model(x)
        return output

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = nn.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decodeur(nn.Module):
    def __init__(self,lattent_dim):
        super().__init__()
        self.model = nn.Sequential(
             nn.Upsample(scale_factor=2),nn.Conv2d(lattent_dim,64,kernel_size=5,padding="same"),nn.BatchNorm2d(64),nn.ReLU(),
             nn.Upsample(scale_factor=2),nn.Conv2d(64,32,kernel_size=5,padding="same"),nn.BatchNorm2d(32),nn.ReLU(),
             nn.Upsample(scale_factor=2),nn.Conv2d(32,1,kernel_size=5,padding="same"),nn.Softplus(),nn.Dropout2d(p=0.2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Variationnal_Auto_encodeur(nn.Module):
    def __init__(self,Decodeur,Encodeur,lattent_dim,audio_spec=True):
        super().__init__()
        self.Decodeur=Decodeur(lattent_dim)
        self.Encodeur=Encodeur(lattent_dim)
        self.audio_spec=audio_spec
     
    def forward(self, x):
        mid_rep= self.Encodeur(x)
        output=self.Decodeur(mid_rep)
        if self.audio_spec==False:
            spectro_inv=GriffinLim(n_fft=1023,hop_length=250,power=2,normalized=True)
            output=spectro_inv(output)
        
        return output,mid_rep    


class Auto_encodeur(nn.Module):
    def __init__(self,Decodeur,Encodeur,lattent_dim,audio_spec=True):
        super().__init__()
        self.Decodeur=Decodeur(lattent_dim)
        self.Encodeur=VariationalEncoder(lattent_dim)
        self.audio_spec=audio_spec
     
    def forward(self, x):
        mid_rep= self.Encodeur(x)
        output=self.Decodeur(mid_rep)
        if self.audio_spec==False:
            spectro_inv=GriffinLim(n_fft=1023,hop_length=250,power=2,normalized=True)
            output=spectro_inv(output)
        
        return output,mid_rep