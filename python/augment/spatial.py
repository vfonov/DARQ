import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math

from geo  import *
from misc import *
from voxel_morph import *

from torch.amp import autocast


class AugSpatial(nn.Module):
    """
    Apply random spatial transforms
    """
    def __init__(self, sample_size, transform_params=None, dtype=torch.float32, device='cpu'):
        super(AugSpatial, self).__init__()
        
        self.transformer = None
        self.integrator = None
        self.nl_smooth = None
        self.sample_size = sample_size
        self.dtype=dtype
        self.device=device

        # create matrices
        self.v2p = create_v2p_matrix(self.sample_size)
        self.p2v = torch.inverse(self.v2p)

        # TODO: use sampling from the metadata instead of hardcoded?
        self.v2w = create_translation_matrix(
            torch.Tensor([-self.sample_size[0]/2, -self.sample_size[1]/2, -self.sample_size[2]/2]))
        
        self.w2v = torch.inverse(self.v2w)

        # 
        self.use_random_transform(transform_params)


    def use_random_transform(self, transform_params):
        self.random_spatial_transform = transform_params

        if self.random_spatial_transform is not None:
            ran_nl_smooth=self.random_spatial_transform.get('nl_smooth',0.0)
            ran_nl_kern=self.random_spatial_transform.get('nl_kern',5)
            
            # generate smoothing kernel (flat)
            if ran_nl_smooth>0.0:
                self.nl_smooth = torch.nn.Conv3d(3,3,ran_nl_kern, padding=ran_nl_kern//2, 
                                                 bias=False, groups=3, dtype=self.dtype)
                self.nl_smooth.weight.data[:,:,:,:,:] = gkern3d(ran_nl_kern,ran_nl_smooth)
                self.nl_smooth.weight.requires_grad = False
                self.nl_smooth.to(self.device)
            # generate integrator and sampler

            self.transformer = SpatialTransformer(self.sample_size, dtype=self.dtype)
            self.transformer.to(self.device)
            self.integrator = VecInt(self.sample_size, 5, dtype=self.dtype)
            self.integrator.to(self.device)
        else:
            self.transformer = None
            self.integrator = None
            self.nl_smooth = None
    
    @torch.autocast(device_type="cuda")
    def forward(self,x):
        if self.random_spatial_transform is not None:

            batch_mri  = x
            batch_size = batch_mri.shape[0]

            # create random affine transform
            ran_rot=self.random_spatial_transform.get('rot',0.0)
            ran_scale=self.random_spatial_transform.get('scale',0.0)
            ran_shift=self.random_spatial_transform.get('shift',0.0)
            ran_shear=self.random_spatial_transform.get('shear',0.0)

            ran_nl_step=self.random_spatial_transform.get('nl_step',10)
            ran_nl_mag=self.random_spatial_transform.get('nl_mag',0.0)
            ran_nl_smooth=self.random_spatial_transform.get('nl_smooth',0.0)
            ran_nl_kern=self.random_spatial_transform.get('nl_kern',5)

            with torch.no_grad():
                # here affine matrix is double (?)
                batch_mri_=[]
                batch_seg_=[]
                batch_dist_=[]

                for b in range(batch_size):
                    _batch_mri = batch_mri[b:b+1,...]

                    _random_rot   = random_rotations(torch.tensor([ran_rot, ran_rot, ran_rot])) if ran_rot>0.0 else torch.eye(4)
                    _random_scale = random_scales(torch.tensor([ran_scale, ran_scale, ran_scale])) if ran_scale>0.0 else torch.eye(4)
                    _random_shift = random_translations(torch.tensor([ran_shift, ran_shift, ran_shift]))  if ran_shift>0.0 else torch.eye(4)
                    _random_shear = random_shears(torch.tensor([ran_shear,]*6))  if ran_shear>0.0 else torch.eye(4)

                    _random_affine = _random_shift @ _random_shear @ _random_scale @ _random_rot

                    _full_affine = self.v2p @ self.w2v @ _random_affine @ self.v2w @ self.p2v

                    _random_affine_grid = grid_to_flow( affine2grid(_full_affine, _batch_mri.shape, 
                                                    device=_batch_mri.device, dtype=self.dtype),
                                                    _batch_mri.shape[2:])
                    
                    # generate random nonlinear transform
                    if ran_nl_step>0.0 and ran_nl_mag>0.0:

                        _random_nl_grid_lr = torch.rand(1,3,math.ceil(_batch_mri.shape[2]/ran_nl_step),
                                                            math.ceil(_batch_mri.shape[3]/ran_nl_step),
                                                            math.ceil(_batch_mri.shape[4]/ran_nl_step),
                                                        dtype=torch.float, 
                                                        device=_random_affine_grid.device)*ran_nl_mag*2-ran_nl_mag

                        # need to put spatial dimension to the last place
                        _random_nl_grid_hr = F.interpolate(_random_nl_grid_lr, 
                                                        size=_batch_mri.shape[2:], 
                                                        mode='trilinear', align_corners=False).\
                                                to(self.dtype).cuda()
                        
                        # apply smoothing 
                        # and put spatial dimension to the last place
                        if ran_nl_smooth>0.0 and ran_nl_kern>0:
                            _random_nl_grid_hr = self.nl_smooth.forward(_random_nl_grid_hr)
                        
                        # make it diffeomorphic
                        _random_nl_grid_hr = self.integrator.forward(_random_nl_grid_hr)

                        # concatenate with affine
                        _random_grid = self.transformer.forward(_random_nl_grid_hr, _random_affine_grid) + _random_affine_grid
                    else:
                        _random_grid = _random_affine_grid
                    
                    _batch_mri = self.transformer.forward(_batch_mri, _random_grid)
                    batch_mri_.append(_batch_mri)

                # now put it all together
                batch_mri = torch.cat(batch_mri_, dim=0)

        return batch_mri
