# -*- coding: utf-8 -*-

#os.chdir('C:/Users/vppit/Desktop/Sorbonne/M_S3/IM/MultiEntVAE')
#os.chdir('../')

import torch
import torchaudio.transforms as T
from argparse import ArgumentParser

from src.VAE import *
from src.dataset_loader import *
from src.plots import *
from src.utils import *
from src.preprocessing import *

starting_time = datetime.now().strftime("%d-%m-%Y_%H_%M")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no-log', action="store_false")
parser.add_argument('--no-save', action="store_false")
parser.add_argument('--load-recent', action="store_true", help="Load most recent model")
parser.add_argument('--load', type=str, default="", help="Load specified model")
parser.add_argument('--no-train', action="store_false", help="No training")

args = parser.parse_args()

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
log = args.no_log
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
    func2module(lambda x:x/x.max())
).to(device)

training_file = "/data/atiam_ml_mvae/nsynth-train"
validation_file = "/data/atiam_ml_mvae/nsynth-valid"

train_dataloader, valid_dataloader = NSYNTH_give_dataloader(training_file,validation_file,batch_size=batch_size,device=device,transform=transform)

training_norm, training_denorm = find_normalizer(train_dataloader,"train")
valid_norm, valid_denorm = find_normalizer(valid_dataloader,'valid')

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
    model = VAE(in_channels=128, hidden_dim = 512, latent_dims=256, beta = beta).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters : {n_params:}")
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir) if log else None

if save:
    saving_model_file = os.path.join("results",f"VAE_{starting_time}")
    torch.save(model,saving_model_file)
else : saving_model_file =""

#------------------------------------------------Training model-----------------------------------------------------------------------
if train:
    print("Training model")
    model= train_VAE(model, train_dataloader, epochs, lr,device,writer, training_norm,saving_model_file)    


#------------------------------------------------Writing tensorboard-----------------------------------------------------------------------
if log:
    tensorboard_writer(model,valid_dataloader,writer,nn.Sequential(valid_denorm,inverse_transform),valid_norm,args)
