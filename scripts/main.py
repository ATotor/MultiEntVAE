# -*- coding: utf-8 -*-


import os

#os.chdir('C:/Users/vppit/Desktop/Sorbonne/M_S3/IM/MultiEntVAE')
#os.chdir('../')

import torch
import torchaudio.transforms as T
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from src.VAE import *
from src.dataset_loader import *
from src.plots import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no-disp', action="store_false")
parser.add_argument('--no-save', action="store_false")
parser.add_argument('--load', action="store_true", help="Load model instead of training")
args = parser.parse_args()

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
disp = args.no_disp
save = args.no_save
load = args.load
#train_dataloader, test_dataloader =  MNIST_give_dataloader(batch_size=batch_size)

transform = T.MelSpectrogram(sample_rate=16000,n_mels=256,n_fft=2048,norm='slaney').to(device=device)
inverse_transform = nn.Sequential(
    T.InverseMelScale(sample_rate=16000,n_mels=256,n_stft=2048 // 2 + 1,norm="slaney"),
    T.GriffinLim(n_fft=2048)
).to(device=device)

train_dataloader, test_dataloader = NSYNTH_give_dataloader(root="data\\nsynth-test",batch_size=batch_size,device=device,transform=transform)

model = VAE(in_channels=256).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters : {n_params:}")
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir) if disp else None

if load is False :
    print("Training model")
    model, loss = train_VAE(model, train_dataloader, epochs, lr,device,writer)    
else:
    print("Loading model")
    model = load_model(find_most_recent_VAE())
    model = model.to(device)
    loss = None
        
if save:
    save_model(model)
    #if loss:    save_loss(loss)

if disp:
    #if loss :     disp_loss(loss)
    #disp_MNIST_example(model, test_dataloader)
    tensorboard_writer(model,test_dataloader,writer,inverse_transform,device)
