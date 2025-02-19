import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math

from .geo  import *
from .misc import *
from .voxel_morph import *

class AugNu(nn.Module):
    """
    Apply random non-uniformity field
    """
    def __init__(self, sample_size, nu_params=None, dtype=torch.float32, device='cpu'):
        super(AugNu, self).__init__()
        self.sample_size = sample_size
        self.dtype = dtype
        self.device = device

        self.use_random_nu(nu_params)


    def use_random_nu(self, nu_params):
        self.random_nu = nu_params
        if self.random_nu is not None:
            ran_nu_smooth=self.random_nu.get('smooth',0.0)
            ran_nu_kern=self.random_nu.get('kern',7)
            ran_nu_channels=self.random_nu.get('channels',1)
            # generate smoothing kernel (flat)
            if ran_nu_smooth>0.0 and ran_nu_kern>0:
                self.nu_smooth = torch.nn.Conv3d(ran_nu_channels,ran_nu_channels,ran_nu_kern, padding=ran_nu_kern//2, bias=False, groups=1)
                self.nu_smooth.weight.data[:,:,:,:,:] = gkern3d(ran_nu_kern,ran_nu_smooth)
                self.nu_smooth.weight.requires_grad = False
                self.nu_smooth.to(self.device)
        else:
            self.ran_nu_smooth = None

    @torch.autocast(device_type="cuda")
    def forward(self,x):
        if self.random_nu is not None:
            batch_mri = x['img']

            ran_nu_smooth=self.random_nu.get('smooth',0.0)
            ran_nu_kern=self.random_nu.get('kern',7)
            ran_nu_channels=self.random_nu.get('channels',1)
            ran_nu_mag=self.random_nu.get('mag',0.0)
            ran_nu_step=self.random_nu.get('step',20)
            
            if ran_nu_mag>0.0:
                with torch.no_grad():
                    # create random non-uniformity field
                   
                    _random_nu_grid_lr = torch.rand(1,ran_nu_channels,
                                                    math.ceil(batch_mri.shape[2]/ran_nu_step),
                                                    math.ceil(batch_mri.shape[3]/ran_nu_step),
                                                    math.ceil(batch_mri.shape[4]/ran_nu_step), 
                                                    dtype=torch.float, device=batch_mri.device) * ran_nu_mag*2-ran_nu_mag
                    # apply smoothing 
                    if ran_nu_smooth>0.0 and ran_nu_kern>0:
                        _random_nu_grid_lr = self.nu_smooth.forward(_random_nu_grid_lr)
                    
                    # need to put spatial dimension to the last place
                    _random_nu_grid_hr = F.interpolate(_random_nu_grid_lr, 
                            size=batch_mri.shape[2:], mode='trilinear', align_corners=False).\
                            to(dtype=self.dtype)
                    
                    # convert to exponent scale
                    _random_nu_grid_hr.exp_()
                    batch_mri *= _random_nu_grid_hr/_random_nu_grid_hr.mean([2,3,4], keepdim=True)
            x['img'] = batch_mri
        return x
            