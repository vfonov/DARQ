import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


class AugCast(nn.Module):
    """
    Simulate intensities using primitive MRI simulator
    """
    def __init__(self, sample_size, dtype=torch.float32, device='cpu'):
        super(AugCast, self).__init__()
        self.sample_size = sample_size
        self.dtype = dtype
        self.device = device

    @torch.autocast(device_type="cuda")
    def forward(self,x):
        return x.to(self.dtype)
            