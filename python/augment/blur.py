import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


from .geo  import *
from .misc import *
from .voxel_morph import *


class AugBlur(nn.Module):
    """
    Apply bluring
    """
    def __init__(self, sample_size, blur_params=None, dtype=torch.float32, device='cpu'):
        super(AugBlur, self).__init__()
        self.sample_size = sample_size
        self.dtype = dtype
        self.device = device
        self.use_blur(blur_params)


    def use_blur(self, blur_params):
        self.blur_params = blur_params
        if self.blur_params is not None:
            # expect array of [x,y,z] for thickness
            blur_smooth=self.blur_params.get('smooth',0.0)
            blur_kern=self.blur_params.get('kern',7)
            blur_channels=self.blur_params.get('channels',1)
            # generate smoothing kernel (flat)
            if blur_smooth>0.0 and blur_kern>0:
                self.blur = torch.nn.Conv3d(blur_channels,blur_channels,blur_kern, padding=blur_kern//2, bias=False, groups=1)
                self.blur.weight.requires_grad = False
                self.blur.weight.data[:,:,:,:,:] = gkern3d(blur_kern,blur_smooth)
                self.blur.to(self.device)
        else:
            self.blur = None

    @torch.autocast(device_type="cuda")
    def forward(self,x):
        if self.blur is not None:
            batch_mri = x
            batch_mri = self.blur.forward(batch_mri)
            x = batch_mri
        return x
            