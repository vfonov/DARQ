import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


from .geo  import *
from .misc import *
from .voxel_morph import *

class AugThickSlices(nn.Module):
    """
    Emulate thick slices
    """
    def __init__(self, sample_size, thick_params=None, dtype=torch.float32, device='cpu'):
        super(AugThickSlices, self).__init__()
        self.sample_size = sample_size
        self.dtype  = dtype
        self.device = device
        self.use_random_thick(thick_params)

    def use_random_thick(self, thick_params):
        self.thick_params = thick_params
        if self.thick_params is not None:
            # expect array of [x,y,z] for thickness
            self.thickness=self.thick_params.get('thickness',[1,1,1])
            channels=self.thick_params.get('channels',1)
            # generate smoothing kernel(s) (flat)
            self.smooth=[None, None, None]
            for i in range(3):
                self.smooth[i]=[]
                for j in range(2,self.thickness[i]+1):
                    # generate sizes
                    ks=tuple(  j + (1-j%2)     if k==i else 1 for k in range(3))
                    # padding 
                    ps=tuple( (j + (1-j%2))//2 if k==i else 0 for k in range(3))
                    # generate kernel
                    self.smooth[i].append(torch.nn.Conv3d(channels, channels, ks, padding=ps, bias=False, groups=1, padding_mode='reflect'))
                    self.smooth[i][-1].weight.requires_grad = False

                    if (j%2) == 1:
                        # flat kernel
                        self.smooth[i][-1].weight.view(-1,j)[...] = 1.0
                    else:
                        # triangular or trapezoid kernel
                        self.smooth[i][-1].weight.view(-1,j + (1-j%2))[...   ] = 1.0
                        self.smooth[i][-1].weight.view(-1,j + (1-j%2))[:,0:2 ] = 0.5
                        self.smooth[i][-1].weight.view(-1,j + (1-j%2))[:,j-1:] = 0.5

                    self.smooth[i][-1].to(self.device)
            
        else:
            self.smooth = None

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        if self.smooth is not None:
            batch_mri = x
            # apply coordinate-wise smoothing
            for i in range(3):
                if self.thickness[i]>1:
                    with torch.no_grad():
                        j=int(torch.randint(1, self.thickness[i]+1,(1,)))
                        # randomly choose smoothing kernel
                        if j>1:
                            batch_mri = self.smooth[i][j-2].forward(batch_mri)
        return batch_mri
            