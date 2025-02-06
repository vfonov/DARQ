import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


from geo  import *
from misc import *
from voxel_morph import *

from torch.amp import autocast

class AugNoise(nn.Module):
    """
    Apply various noise to the input
    """
    def __init__(self,
                 mult_noise=0.0, 
                 bias_noise=0.0,
                 add_noise=0.0,
            ):
        super(AugNoise, self).__init__()
        self.set_parameters(mult_noise,bias_noise,add_noise)
        
    def set_parameters(self,mult_noise=0.0,
                        bias_noise=0.0,add_noise=0.0,
                        corrupt_noise=0.0,channel_drop=0.0,
                        clone_input=False ):
        self.mult_noise = mult_noise
        self.bias_noise = bias_noise
        self.add_noise = add_noise
    
    @torch.autocast(device_type="cuda")
    def forward(self,x):
        # global multiplicative random value
        mri = x

        sz     = mri.size()
        bs     = sz[0]

        if self.mult_noise>0.0:
            _noise_mult = torch.empty(bs, 1, 1, 1, 1, device=mri.device, dtype=mri.dtype)\
                .normal_(1.0, self.mult_noise)
            mri.mul_(_noise_mult)

        # global bias random value
        if self.bias_noise>0.0 :
            _noise_bias = torch.empty(bs, 1, 1, 1, 1, device=mri.device, dtype=mri.dtype)\
                .normal_(0.0, self.bias_noise)
            mri.add_(_noise_bias)

        # per-voxel random gaussian noise
        if self.add_noise>0.0:
            # sigma = self.add_noise * self.add_noise_gamma**(batch_no//self.add_noise_step)
            _noise_add = torch.empty_like(mri).normal_(0.0, self.add_noise )
            mri.add_(_noise_add)

        # per-voxel corruption (set to 0)
        if self.corrupt_noise is not None and self.corrupt_noise>0.0:
            _noise_corrupt = torch.lt(torch.empty_like(mri).uniform_(), self.corrupt_noise)
            mri[_noise_corrupt] = 0.0

        return mri
