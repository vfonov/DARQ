import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from data.geo  import *
from data.misc import *



class AugSlices(nn.Module):
    """
    Extract central x,y,z slices from the input
    """
    def __init__(self, sample_size, patch_size ):
        super(AugSlices, self).__init__()
        self.sample_size = sample_size
        # calculate padding or cropping
        crop=[0,0,0]
        pad=[0,0,0]

        for i in range(3):
            if sample_size[i] > patch_size[i]:
                crop[i] = (sample_size[i] - patch_size[i])//2
            elif sample_size[i] < patch_size[i]:
                pad[i] = (patch_size[i] - sample_size[i])//2

        self.crop = crop
        self.pad = pad
        self.cx = [sample_size[0]//2, sample_size[1]//2, sample_size[2]//2]
        self.sample_size = sample_size
    
    @torch.autocast(device_type="cuda")
    def forward(self,x):

        slices = [ x[:,self.cx[0], self.crop[1]:self.sample_size[1]-self.crop[1]*2, self.crop[2]:self.sample_size[2]-self.crop[2]*2], # x
                   x[:,self.crop[0]:self.sample_size[0]-self.crop[0]*2,self.cx[1],self.crop[2]:self.sample_size[2]-self.crop[2]*2], # y
                   x[:,self.crop[0]:self.sample_size[0]-self.crop[0]*2,self.crop[1]:self.sample_size[1]-self.crop[1]*2,self.cx[2]]]
        # apply padding
        slices = [ F.pad(slices[0], (self.pad[2],self.pad[2],self.pad[1],self.pad[1]), "replicate"),
                   F.pad(slices[1], (self.pad[2],self.pad[2],self.pad[0],self.pad[0]), "replicate"),
                   F.pad(slices[2], (self.pad[1],self.pad[1],self.pad[0],self.pad[0]), "replicate"),
                   ]

        return torch.concat(slices,1)