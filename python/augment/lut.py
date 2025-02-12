import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


def interp1d_pytorch_batched(x, y, x_new):
    """
    Linearly interpolate data points in PyTorch with support for batched x_new.

    Parameters:
    x (torch.Tensor): 1D tensor of known x-coordinates (must be sorted).
    y (torch.Tensor): 1D tensor of known y-values corresponding to x.
    x_new (torch.Tensor): Tensor of new x-coordinates to interpolate, with shape (batch_size, n_points).

    Returns:
    torch.Tensor: Interpolated values at x_new, with shape (batch_size, n_points).

    Written by ChatGPT, fixed by me
    """
   
    # Expand x for broadcasting over batch dimensions
    #x_expanded = x.unsqueeze(0)  # Shape: (1, len(x))

    # Clamp x_new values within the range of x
    #x_new_clamped = torch.clamp(x_new, min=x.min().item(), max=x.max().item())
    # Get indices for the points in x that are just below and above each x_new in the batch
    idx = torch.searchsorted(x, x_new, right=True) - 1
    idx = torch.clamp(idx, 0, x.shape[-1] - 2)

    # Gather the x and y values at the selected indices for each batch
    x0, x1 = torch.take_along_dim(x,idx,dim=1), torch.take_along_dim(x,idx + 1,dim=1)
    y0, y1 = torch.take_along_dim(y,idx,dim=1), torch.take_along_dim(y,idx + 1,dim=1)
    
    # Perform linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    y_new = y0 + slope * (x_new - x0)
    
    return y_new


class AugLUT(nn.Module):
    """
    Apply random Look-Up-Table (LUT) to the input, all inputs assumed to be in [0,1]
    """
    def __init__(self, n_bins=20, strength=1.0):
        super(AugLUT, self).__init__()
        self.n_bins = n_bins
        self.strength = strength


    @torch.autocast(device_type="cuda")
    def forward(self,x):
        if self.strength > 0.0:
            mri    = x
            sz     = mri.shape
            bs     = sz[0]

            with torch.no_grad():

                ran_x = torch.linspace(0,1,self.n_bins,device=mri.device, dtype=mri.dtype).unsqueeze(0).expand(bs,self.n_bins)

                ran_y = torch.rand((bs,self.n_bins),   device=mri.device, dtype=mri.dtype)
                lin_y = torch.linspace(0,1,self.n_bins,device=mri.device, dtype=mri.dtype).unsqueeze(0).expand(bs,self.n_bins)

                y = ran_y*self.strength + lin_y * (1.0-self.strength)

                # normalize y to [0,1]
                y = (y-y.amin(dim=1,keepdim=True))/(y.amax(dim=1,keepdim=True)-y.amin(dim=1,keepdim=True) + 1e-5)

                # apply random lut
                _out = interp1d_pytorch_batched(ran_x, y, mri.reshape(bs,-1)).reshape(mri.shape)
            
            return _out
        else: # bypass
            return x 
