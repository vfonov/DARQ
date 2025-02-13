import torch
from torch import nn
from augment import *



class AugSqueeze(nn.Module):
    """
    Remove extra dimension (for pre-sliced data)
    """
    def __init__(self,autonorm=True):
        super(AugSqueeze, self).__init__()

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        return torch.squeeze(x,1)

class AugClamp(nn.Module):
    """
    Remove NaNs and infinities, clamp to 0-1
    """
    def __init__(self,autonorm=True):
        super(AugClamp, self).__init__()

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        return torch.nan_to_num(x,nan=0.0,posinf=1.0,neginf=0.0).clip(0.0,1.0)



def create_augment_model(params, dataset,
        device='cuda',dtype=torch.float16,
        testing=False):
    
    sample_size=dataset.sample_size
    patch_size=params['patch_size']
    noise=params['noise']
    lut=params['lut']
    slices=params['slices']
    nu_strength=params['nu_strength']
    nl_mag=params['nl_mag']
    thickness=params['thickness']

    print(f"{sample_size=},{patch_size=},{slices=}")

    if slices: # simpler dataset, only add noise and LUT
        if testing:
            mod=nn.Sequential(
                AugClamp(),
                AugSqueeze()
            )
        else:
            mod=nn.Sequential(
                AugLUT(n_bins=20, strength=lut),
                AugNoise(add_noise=noise),
                Autonorm(),
                AugSqueeze()
                )
    else:
        if testing:
            mod=nn.Sequential(
                Autonorm(),
                AugSlices(sample_size,patch_size),
            )
        else:
            mod=nn.Sequential(
                Autonorm(),
                AugNu(sample_size, dict(smooth=7.0, kern=7, mag=nu_strength,step=20), dtype=dtype, device=device),
                AugLUT(n_bins=20, strength=lut),
                AugNoise(add_noise=noise),
                AugThickSlices(sample_size,dict(thickness=[thickness,thickness,thickness]),dtype=dtype,device=device) ,
                AugSpatial(sample_size, 
                    dict(rot=0.01,scale=0.01,shift=0.0,shear=0.0,
                        nl_step=10,
                        nl_mag=nl_mag,
                        nl_smooth=0.5,nl_kern=5),dtype=dtype,device=device),
                Autonorm(),
                AugSlices(sample_size,patch_size)
            )
    mod=mod.to(device)
    print("Augmentation modules:",mod)
    return mod