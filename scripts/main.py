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
from src.utils import *
from src.preprocessing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no-disp', action="store_false")
parser.add_argument('--no-save', action="store_false")
parser.add_argument('--load-recent', action="store_true", help="Load most recent model instead of training")
parser.add_argument('--load', type=str, default="", help="Load specified model instead of training")

args = parser.parse_args()

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
disp = args.no_disp
save = args.no_save
load_recent = args.load_recent
load = args.load
beta = args.beta

transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=16000,n_mels=256,n_fft=2048,norm='slaney'),
    #Normalize()
).to(device)
inverse_transform = nn.Sequential(
    T.InverseMelScale(sample_rate=16000,n_mels=256,n_stft=2048 // 2 + 1,norm="slaney"),
    #T.GriffinLim(n_fft=2048),
    librosa_GriffinLim(n_fft=2048),
    #Normalize(),
).to(device)


train_dataloader, test_dataloader, train_spec_normalizer, train_spec_denormalizer, test_spec_normalizer, test_spec_denormalizer = NSYNTH_give_dataloader(root="data\\nsynth-test",batch_size=batch_size,device=device,transform=transform)

model = VAE(in_channels=256,hidden_dim = 128, latent_dims=256, beta = beta).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters : {n_params:}")
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir) if disp else None

if not load and load_recent is False:
    print("Training model")
    model, loss = train_VAE(model, train_dataloader, epochs, lr,device,writer, Normalize())    

elif load_recent:
    print("Loading most recent model")
    model = load_model(find_most_recent_VAE())
    model = model.to(device)
    loss = None
elif load:
    print("Loading model" +load)
    model = load_model(load)
    model = model.to(device)
    loss = None    

if save:
    save_model(model)
    #if loss:    save_loss(loss)

if disp:
    #if loss :     disp_loss(loss)
    #disp_MNIST_example(model, test_dataloader)
    tensorboard_writer(model,test_dataloader,writer,nn.Sequential(nn.Identity(),inverse_transform),Normalize(),args)
