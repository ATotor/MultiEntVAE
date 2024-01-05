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
import torch.nn.functional as F
import pandas as pd

class NsynthDataset(Dataset):
    def __init__(self,json_file,sound_dir,transform = Spectrogram(n_fft=511,hop_length=125,power=2,normalized=True)):
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
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),   
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Decodeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
             nn.Upsample(scale_factor=2),nn.Conv2d(128,64,kernel_size=5,padding="same"),nn.BatchNorm2d(64),nn.ReLU(),
             nn.Upsample(scale_factor=2),nn.Conv2d(64,32,kernel_size=5,padding="same"),nn.BatchNorm2d(32),nn.ReLU(),
             nn.Upsample(scale_factor=2),nn.Conv2d(32,1,kernel_size=5,padding="same"),nn.Softplus(),nn.Dropout2d(p=0.2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

   
class Auto_encodeur(nn.Module):
    def __init__(self,Decodeur,Encodeur,audio_spec=True):
        super().__init__()
        self.Decodeur=Decodeur
        self.Encodeur=Encodeur
        self.audio_spec=audio_spec
     
    def forward(self, x):
        mid_rep= self.Encodeur(x)
        output=self.Decodeur(mid_rep)
        if self.audio_spec==False:
            spectro_inv=GriffinLim(n_fft=511,hop_length=125,power=2,normalized=True)
            output=spectro_inv(output)
        
        return output,mid_rep
    

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1=nn.Sequential(nn.Conv2d(1,2,kernel_size=5,stride=2,padding=2),nn.ReLU())#128
        self.enc2=nn.Sequential(nn.Conv2d(2,4,kernel_size=5,stride=2,padding=2),nn.ReLU())#64
        self.enc3=nn.Sequential(nn.Conv2d(4,8,kernel_size=5,stride=2,padding=2),nn.ReLU())#32
        self.enc4=nn.Sequential(nn.Conv2d(8,16,kernel_size=5,stride=2,padding=2),nn.ReLU())#16
        self.enc5=nn.Sequential(nn.Conv2d(16,32,kernel_size=5,stride=2,padding=2),nn.ReLU())#8
        self.enc6=nn.Sequential(nn.Conv2d(32,64,kernel_size=8,stride=1,padding=0),nn.ReLU())#4
        
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, 200)
        self.fc_log_var = nn.Linear(128, 200)
        self.fc2 = nn.Linear(200, 64)
        # decoder 
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=4),
        nn.Conv2d(64,32,kernel_size=5,padding="same"),nn.BatchNorm2d(32),nn.ReLU())
        
        self.dec2 =  nn.Sequential(nn.Upsample(scale_factor=4),
        nn.Conv2d(32,16,kernel_size=5,padding="same"),nn.BatchNorm2d(16),nn.ReLU())
        
        self.dec3 =  nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(16,8,kernel_size=5,padding="same"),nn.BatchNorm2d(8),nn.ReLU())
        
        self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(8,4,kernel_size=5,padding="same"),nn.BatchNorm2d(4),nn.ReLU())
        
        self.dec5 =  nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(4,2,kernel_size=5,padding="same"),nn.BatchNorm2d(2),nn.ReLU())
        
        self.dec6 =  nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(2,1,kernel_size=5,padding="same"),nn.ReLU())
        
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        
        batch, _, _, _ = x.shape
        x=F.avg_pool2d(x,1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        x = self.dec1(z)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        reconstruction = self.dec6(x)
        
        return reconstruction, mu, log_var