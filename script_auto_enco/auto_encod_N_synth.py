# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:26:44 2023

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram,GriffinLim
from argument_parser_N_synth import parse_arguments
from torch.utils.tensorboard import SummaryWriter
from model_dataset import NsynthDataset,Encodeur,Decodeur,Auto_encodeur
import torchaudio

    
def train_model(config):
    num_epochs=config["num_epochs"]
    lr=config["learning_rate"]
    batch_size=config["batch_size"]
    
    model_path=config["model_dir"]
    model_name=config["model_name"]
    tensorboard_dir=config["tensorboard_dir"]
    
    sound_dir_train=config["train_sound_dir"]
    sound_dir_test=config["test_sound_dir"]
    
    json_file_train=config["train_json_file"]
    json_file_test=config["test_json_file"]
    
    writer = SummaryWriter(tensorboard_dir)

        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    train_data_set=NsynthDataset(json_file_train,sound_dir_train)
    train_dataloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    
    test_data_set=NsynthDataset(json_file_test,sound_dir_test)
    test_dataloader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)
    
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
          print("CUDA is available! Training on GPU...")
    else:
          print("CUDA is not available. Training on CPU...")
    enco=Encodeur(128).to(device)
    deco=Decodeur(128).to(device)
    auto_enco=Auto_encodeur(deco, enco,128).to(device)
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(auto_enco.parameters(),lr=lr)
    dataiter = iter(train_dataloader)
    spec, features = next(dataiter)
    spec=spec.to(device)
    writer.add_graph(auto_enco, spec)
    n_params = sum(p.numel() for p in auto_enco.parameters())
    print(f"nombre de paramètres du model: {n_params}")
    auto_enco.train()
    running_loss=0
    #%%'
    best_vloss = 1_000_000.
    for epoch in range(num_epochs):
        auto_enco.train()
        print('epoch {}'.format(epoch))
        for n, (X_sample, _) in enumerate(train_dataloader):
           auto_enco.zero_grad()
           X_sample=X_sample.to(device)
           X_recons,_=auto_enco.forward(X_sample)
           X_recons=X_recons.to(device)
           loss=criterion(X_sample,X_recons)+auto_enco.encoder.kl
           running_loss+=loss.item()
           loss.backward()
           optimizer.step()
           if n == batch_size - 1:
              avg_loss = running_loss /batch_size  # loss per batch
              tb_x = epoch * len(train_dataloader) + n + 1
              running_loss = 0.
        running_vloss = 0.0
        auto_enco.eval()
        with torch.no_grad():
            for i, (X_sample_test,_) in enumerate(test_dataloader):
               X_sample_test=X_sample_test.to(device)
               X_recons_test,_=auto_enco.forward(X_sample_test)
               vloss = criterion(X_recons_test, X_sample_test)
               running_vloss += vloss
            avg_vloss = running_vloss / (i + 1)
        print('avg_loss train: {} avg_loss valid: {}'.format(avg_loss, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path_2 =model_path+"/"+'model_{}_{}_loss_val_{}'.format(model_name, epoch,round(best_vloss.item(),3))
            torch.save(auto_enco.state_dict(), model_path_2)
    writer.flush()
    
if __name__ == "__main__":
    train_model(parse_arguments())