import torch
import torch.nn as nn
import torch.nn.functional as nnf


class MaxAutonorm(nn.Module):
    """
    Normalize MRI data between 0 and 1
    """
    def __init__(self, autonorm=True):
        super(MaxAutonorm, self).__init__()

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        batch_mri = x
        batch_mri /= batch_mri.reshape(batch_mri.shape[0], batch_mri.shape[1], -1).float().max(dim=2)[0].reshape(batch_mri.shape[0], batch_mri.shape[1], 1, 1, 1)
        return batch_mri