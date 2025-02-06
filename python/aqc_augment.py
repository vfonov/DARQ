import torch
from torch import nn
from augment import *

def create_augment_model(params, dataset,device='cuda',dtype=torch.float16,testing=False):
    sample_size=dataset.sample_size
    patch_size=params['patch_size']
    print(f"{sample_size=},{patch_size=}")

    if testing:
        mod=nn.Sequential(
            Autonorm(),
            AugSlices(sample_size,patch_size)
        )
    else:
        mod=nn.Sequential(
            Autonorm(),
            AugNu(sample_size, dict(smooth=7.0,kern=7,mag=2.0,step=20), dtype=dtype, device=device),
            AugLUT(n_bins=20, strength=0.2),
            AugNoise(add_noise=0.1),
            AugThickSlices(sample_size,dict(thickness=[3,3,3]),dtype=dtype,device=device) ,
            AugSpatial(sample_size, 
                dict(rot=0.01,scale=0.01,shift=0.0,shear=0.0,
                    nl_step=10,
                    nl_mag=0.01,
                    nl_smooth=0.1,nl_kern=5),dtype=dtype,device=device),
            Autonorm(),
            AugSlices(sample_size,patch_size)
        )

    print("Augmentation modules:",mod)
    return mod