import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio

class NsynthDataset(Dataset):
    def __init__(self,json_file,sound_dir,transform = torchaudio.transforms.Spectrogram(n_fft=2048)):
        super().__init__()
        self.json = pd.read_json(json_file,orient="index")
        self.sound_dir=sound_dir
        self.transform = lambda x : transform(x)#[0] En fonction de s'il y a 1 channel ou nfft/2 channels

    def __len__(self):
        return len(self.json)
    
    def __getitem__(self, idx):
        note_str = self.json['note_str'].iloc[idx]
        waveform, sampling_rate = torchaudio.load(os.path.join(self.sound_dir,note_str+".wav"))
        spectrogram = self.transform(waveform)
        return spectrogram, self.json['pitch'].iloc[idx]

