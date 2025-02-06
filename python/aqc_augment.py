import torch
from torch import nn
from augment import *

def create_augment_model(params, dataset,device='cuda',dtype=torch.float16):
    sample_size=dataset.sample_size
    patch_size=params['patch_size']

    mod=nn.Sequential(
        Autonorm(),
        AugNu(sample_size, dict(smooth=7.0,kern=7,mag=2.0,step=20), dtype=dtype, device=device),
        AugLUT(n_bins=20, strength=0.2),
        AugNoise(add_noise=0.1),
        AugThickSlices(sample_size,dict(thickness=[3,3,3]),dtype=dtype,device=device) ,
        AugSpatial(sample_size, dict(rot=15.0,scale=0.1,shift=0.1,shear=0.1,nl_step=10,nl_mag=0.1,nl_smooth=0.1,nl_kern=5),dtype=dtype,device=device),
        AugSlices()
    )

    print("Augmentation modules:",mod)
    return mod