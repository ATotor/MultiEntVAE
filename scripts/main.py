# -*- coding: utf-8 -*-

#os.chdir('C:/Users/vppit/Desktop/Sorbonne/M_S3/IM/MultiEntVAE')
#os.chdir('../')

import torch
import pickle
from torchinfo import summary

import torchaudio.transforms as T
from argparse import ArgumentParser

from src.VAE import *
from src.dataset_loader import *
from src.plots import *
from src.utils import *
from src.preprocessing import *

starting_time = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no-save', action="store_false")
parser.add_argument('--load-recent', action="store_true", help="Load most recent model")
parser.add_argument('--load', type=str, default="", help="Load specified model")
parser.add_argument('--no-train', action="store_false", help="No training")

args = parser.parse_args()

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
save = args.no_save
load_recent = args.load_recent
load = args.load
beta = args.beta
train = args.no_train

#------------------------------------------------Spectral paramaeters----------------------------------------------------------------------
nfft = 1024
n_mels = 128
hop_length = nfft //4
Fe = 16000


transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=Fe,n_mels=n_mels,n_fft=nfft, hop_length=hop_length,norm='slaney'),
    func2module(padtime),
    func2module(torch.log1p),
    #Normalize()
).to(device)
inverse_transform = nn.Sequential(
    func2module(torch.expm1),
    T.InverseMelScale(sample_rate=Fe,n_mels=n_mels,n_stft=nfft // 2 + 1,norm="slaney"),
    #T.GriffinLim(n_fft=2048),
    librosa_GriffinLim(n_fft=nfft, hop_length=hop_length),
    Normalize('max1')
).to(device)

training_file = "/data/atiam_ml_mvae/nsynth-train"
validation_file = "/data/atiam_ml_mvae/nsynth-valid"
testing_file = "data/nsynth-test"

train_dataloader, valid_dataloader = NSYNTH_give_dataloader(training_file,validation_file,batch_size=batch_size,device=device,transform=transform)

training_norm, training_denorm = find_normalizer(train_dataloader,"test")
valid_norm, valid_denorm = find_normalizer(valid_dataloader,'test')

#------------------------------------------------Creating model-----------------------------------------------------------------------
if load_recent:
    print("Loading most recent model")
    model = load_model(find_most_recent_VAE())
    model = model.to(device)
elif load:
    print("Loading model" +load)
    model = load_model(load)
    model = model.to(device)
else:
    model = VAE(in_channels=128, 
                hidden_dim = 512, 
                latent_dims=256, 
                beta = beta).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters : {n_params:}")
log_dir = "logs/" + starting_time
writer = SummaryWriter(log_dir)
log_arg(writer,args)
if save:
    saving_model_file = os.path.join("results",f"VAE_{starting_time}")
    torch.save(model,saving_model_file)
else : saving_model_file =""

model_summary = summary(model, input_size=(batch_size,128,128), depth=4)
with open('summary.txt', 'w') as f:
    f.write(str(model_summary))


#------------------------------------------------Training model-----------------------------------------------------------------------
if train:
    print("Training model")
    model = train_VAE(model=model, 
              dataloader=train_dataloader, 
              valid_dataloader=valid_dataloader, 
              valid_norm=valid_norm,
              valid_denorm=valid_denorm,
              inverse_transform=inverse_transform,
              epochs=epochs, 
              lr=lr, 
              device=device,
              writer=writer, 
              spec_normalizer=training_norm,
              saving_model_file=saving_model_file,
              )
    
#------------------------------------------------Creating logs-----------------------------------------------------------------------
tensorboard_writer(model=model,
                    valid_dataloader=valid_dataloader,
                    writer=writer,
                    inverse_transform=nn.Sequential(valid_denorm,inverse_transform),
                    normalizer=valid_norm,
                    batch_size=3,
                    epoch="Final"
                    )