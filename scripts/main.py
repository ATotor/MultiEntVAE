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
    T.GriffinLim(n_fft=nfft,hop_length=hop_length),
    #librosa_GriffinLim(n_fft=nfft, hop_length=hop_length),
    Normalize('max1')
).to(device)
#-----------------------------------------------Dataset construction-----------------------------------------------------------------
minpitch=24
maxpitch=96

training_file = "/data/atiam_ml_mvae/nsynth-train"
validation_file = "/data/atiam_ml_mvae/nsynth-valid"
testing_file = "data/nsynth-test"

train_dataloader, valid_dataloader = NSYNTH_give_dataloader(training_file,validation_file,batch_size=batch_size,device=device,transform=transform,minpitch=minpitch,maxpitch=maxpitch)

training_norm, training_denorm = find_normalizer(train_dataloader,"test")
valid_norm, valid_denorm = find_normalizer(valid_dataloader,'test')

iter_train = iter(train_dataloader)
iter_valid = iter(valid_dataloader)
train_dummy_item1 = next(iter_train)
train_dummy_item2 = next(iter_train)
valid_dummy_item1 = next(iter_valid)
valid_dummy_item2 = next(iter_valid)
#------------------------------------------------Creating model-----------------------------------------------------------------------
if load_recent:
    print("Loading most recent model")
    model = load_model(find_most_recent_VAE())
    model = model.to(device)
elif load:
    print("Loading model " +load)
    model = load_model(load)
    model = model.to(device)
else:
    model = ConVAE(in_channels=train_dummy_item1['x'].shape[1], 
                hidden_dims=[256,512,512], 
                latent_dims=256, 
                beta = beta,
                minpitch=minpitch,
                maxpitch=maxpitch
                ).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters : {n_params:}")

model_summary = summary(model, input_size=train_dummy_item1['x'].shape,verbose=0)
with open('summary.txt', 'w') as f:
    f.write(str(model_summary))

log_dir = "logs/" + starting_time + "_beta=" + str(beta)
writer = SummaryWriter(log_dir)
log_arg(writer,args)
if save:
    saving_model_file = os.path.join("results",f"VAE_{starting_time}")
    torch.save(model,saving_model_file)
else : saving_model_file =""

#------------------------------------------------Training model-----------------------------------------------------------------------
if train:
    print("Training model")
    model = train_VAE(model=model, 
              dataloader=train_dataloader, 
              valid_dummy_item1=valid_dummy_item1, 
              train_dummy_item1=train_dummy_item1,
              valid_dummy_item2=valid_dummy_item2, 
              train_dummy_item2=train_dummy_item2,
              valid_norm=valid_norm,
              valid_denorm=valid_denorm,
              training_norm=training_norm,
              training_denorm=training_denorm,
              inverse_transform=inverse_transform,
              epochs=epochs, 
              lr=lr, 
              device=device,
              writer=writer, 
              saving_model_file=saving_model_file,
              )
    
#------------------------------------------------Creating logs-----------------------------------------------------------------------
tensorboard_writer( model=model,
                    item1=train_dummy_item1, 
                    item2=train_dummy_item2, 
                    dataloader_type = "Training",
                    writer=writer,
                    inverse_transform=nn.Sequential(training_denorm,inverse_transform),
                    normalizer=training_norm,
                    batch_size=None,
                    epoch=epochs
                    )

tensorboard_writer( model=model,
                    item1=valid_dummy_item1, 
                    item2=valid_dummy_item2, 
                    dataloader_type = "Validation",
                    writer=writer,
                    inverse_transform=nn.Sequential(valid_denorm,inverse_transform),
                    normalizer=valid_norm,
                    batch_size=None,
                    epoch=epochs
                    )

