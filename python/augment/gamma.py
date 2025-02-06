import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


from torch.amp import autocast

class AugGamma(nn.Module):
    """
    Apply random gamma function to change the contrast
    """
    def __init__(self,
                 gamma_mag=0.0,
                 clone_input=False,
            ):
        super(AugGamma, self).__init__()
        self.set_parameters(gamma_mag,clone_input)
        
    def set_parameters(self,gamma_mag,
                        clone_input=False ):
        self.gamma_mag = gamma_mag
        self.clone_input = clone_input
    
    @torch.autocast(device_type="cuda")
    def forward(self,x):
        # global multiplicative random value
        mri = x

        sz     = mri.size()
        bs     = sz[0]

        # presever original input
        # to train the denoiser
        
        if self.gamma_mag>0.0:
            _exponent = torch.empty(bs, 1, 1, 1, 1, device=mri.device, dtype=mri.dtype)\
                .normal_(1.0, self.gamma_mag)
            mri.pow_(_exponent)

        return mri
