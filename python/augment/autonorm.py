import torch
import torch.nn as nn

class Autonorm(nn.Module):
    """
    Normalize MRI data between 0 and 1, uisng 99th percentile and minimum
    """
    def __init__(self,autonorm=True):
        super(Autonorm, self).__init__()

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        batch_mri = x
        batch_mri -= batch_mri.amin([2,3,4],keepdim=True)
        batch_mri /= (batch_mri.reshape(batch_mri.shape[0],batch_mri.shape[1],-1).float().quantile(0.99,dim=2)+1e-3).reshape(batch_mri.shape[0],batch_mri.shape[1],1,1,1)
        batch_mri =  torch.nan_to_num(batch_mri,nan=0.0,posinf=1.0,neginf=0.0).clip(0.0,1.0)
        return batch_mri