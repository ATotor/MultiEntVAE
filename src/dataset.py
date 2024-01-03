import os
import torch.nn as nn
from glob import glob
import json
import torch
from torch.utils.data import Dataset
import librosa as li
from src.utils import * 

class NSynth(Dataset):
    def __init__(self, 
                 top_path: str, 
                 *, 
                 device: str | torch.device = "cpu",
                 n_signal: float | None = None, #Will be converted to a power of 2
                 valid_pitch: tuple[int, int] | None = None, 
                 valid_inst: list[str] | None = None, 
                 valid_source: list[str] | None = None,
                 transform: nn.Module | None = None,
        ):
        self.device = device
        self.valid_pitch = valid_pitch
        self.valid_inst = valid_inst
        self.valid_source = valid_source
        files = glob(f'{os.path.abspath(top_path)}/**/*.wav', recursive=True)
        self.filenames = [os.path.basename(f).removesuffix('.wav') for f in files]
        with open(os.path.join(top_path, 'examples.json')) as f:
            self.meta = json.load(f)
        if valid_pitch is not None:
            minpitch, maxpitch = valid_pitch
            files = filter(lambda f: minpitch <= self.meta[f]['pitch'] <= maxpitch, self.filenames)
        if valid_inst is not None:
            files = filter(lambda f: self.meta[f]['instrument_family_str'] in valid_inst, self.filenames)
        if valid_source is not None:
            files = filter(lambda f: self.meta[f]['instrument_source_str'] in valid_source, self.filenames)
        self.files = list(files)
        self.n_signal = next_power_of_2(n_signal)
        self.transform = transform if transform is not None else nn.Identity()
        print(f"There are {len(self.files)} files in the dataset.")
        return
    
    def __len__(self):
            return len(self.files)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, str]:
        fname = self.files[idx]
        _fname = self.filenames[idx]
        audio, _ = li.load(fname, sr=16000, mono=True, duration=self.n_signal / self.meta[_fname]['sample_rate'])
        audio = torch.from_numpy(audio).to(self.device)
        pitch = self.meta[_fname]['pitch']
        return {"x": self.transform(audio), "pitch": pitch, "fname": fname,  "audio" : audio}
    