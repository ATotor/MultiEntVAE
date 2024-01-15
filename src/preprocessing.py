import torch
import torch.nn as nn
import librosa as li
import os
from tqdm import tqdm
from src.utils import *

class librosa_GriffinLim(nn.Module):
    def __init__(self,n_fft=1024, hop_length=256):
        super(librosa_GriffinLim, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
    def forward(self, x):
        return li.griffinlim(x.cpu().numpy(),n_fft=self.n_fft,hop_length=self.hop_length)

def find_normalizer(dataloader,type="train"):
    if not os.path.isfile("min_"+type):
        minval, maxval = float("inf"), float("-inf")
        print(f"Finding {type} normalizer")
        for batch in tqdm(dataloader):
            x = batch["x"]
            minval = min(minval, x.min())
            maxval = max(maxval, x.max())
        torch.save(minval,"min_"+type)
        torch.save(maxval,"max_"+type)
    else:
        minval = torch.load("min_"+type)
        maxval  = torch.load("max_"+type)

    norm = func2module(lambda x: (x - minval) / (maxval - minval))
    denorm = func2module(lambda x: x * (maxval - minval) + minval)

    return norm, denorm

def padtime(x: torch.Tensor) -> torch.Tensor:
    *_, n_sample = x.shape
    n_pad = 2 ** (n_sample - 1).bit_length() - n_sample
    return nn.functional.pad(x, (0, n_pad))    
