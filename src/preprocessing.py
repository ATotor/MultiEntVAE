import torch.nn as nn
import librosa as li

    
class librosa_GriffinLim(nn.Module):
    def __init__(self,n_fft=2048):
        super(librosa_GriffinLim, self).__init__()
        self.n_fft = n_fft
    def forward(self, x):
        return li.griffinlim(x.cpu().numpy(),n_fft=self.n_fft)
    
def find_spec_normalizer(dataloader):
    minval, maxval = float("inf"), float("-inf")
    for batch in dataloader:
        x = batch["x"]
        minval = min(minval, x.min())
        maxval = max(maxval, x.max())
    norm = lambda x: (x - minval) / (maxval - minval)
    denorm = lambda x: x * (maxval - minval) + minval
    return norm, denorm

