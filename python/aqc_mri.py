import os
import math

import torch
import torchvision.transforms.functional as F


def load_minc_images(path,missing_zero=False):
    from minc2_simple import minc2_file 

    if os.path.exists(path):
        input_minc=minc2_file(path)
        input_minc.setup_standard_order()

        w=input_minc.load_complete_volume_tensor()
        sz=w.shape
        w=(w-w.min())/(w.max()-w.min())-0.5

        input_images = [w[sz[0]//2, :, :],
                        w[:, :, sz[2]//2],
                        w[:, sz[1]//2, :]]
            
        # flip, resize and crop
        for i in range(3):
            # 
            _scale = min(256.0/input_images[i].shape[0],
                        256.0/input_images[i].shape[1])
            # vertical flip and resize

            new_sz=[ math.floor(input_images[i].shape[0]*_scale),
                    math.floor(input_images[i].shape[1]*_scale) ]

            dummy=torch.full([1, 256, 256],-0.5)

            dummy[:,(256-new_sz[0])//2:(256-new_sz[0])//2+new_sz[0], 
                    (256-new_sz[1])//2:(256-new_sz[1])//2+new_sz[1]] = \
                F.resize(input_images[i].flip(0).unsqueeze(0),
                        size=new_sz, antialias=True)

            # crop, convert to proper type
            input_images[i]=dummy[:,16:240, 16:240]# +0.5)*255).clamp(0,255).to(torch.uint8)
    else:
        if missing_zero:
            input_images = [torch.full([1, 224, 224],-0.5),
                            torch.full([1, 224, 224],-0.5),
                            torch.full([1, 224, 224],-0.5)]
        else:
            raise NameError(f"File {path} not found")
   
    return input_images


def load_talairach_mgh_images(path):
    import nibabel as nib
    import numpy as np
    from nibabel.affines import apply_affine
    import numpy.linalg as npl

    # conversion from MNI305 space to ICBM-152 space,according to $FREESURFER_HOME/average/mni152.register.dat
    mni305_to_icb152 = np.array( [ [ 9.975314e-01, -7.324822e-03, 1.760415e-02, 9.570923e-01   ], 
                                   [ -1.296475e-02, -9.262221e-03, 9.970638e-01, -1.781596e+01 ],
                                   [ -1.459537e-02, -1.000945e+00, 2.444772e-03, -1.854964e+01 ],
                                   [ 0, 0, 0, 1 ] ] )
    
    icbm152_to_mni305 = npl.inv(mni305_to_icb152)

    img = nib.load(path)
    img_data = img.get_fdata()
    sz=img_data.shape
    
    # transformation from ICBM152 space to the Voxel space in the Freesurfer MNI305 file 
    icbm_to_vox=npl.inv(icbm152_to_mni305 @ img.affine) #

    icbm_origin=np.array([193/2-96, 229/2-132, 193/2-78])
    icbm_origin_x=icbm_origin+np.array([1,0,0])
    icbm_origin_y=icbm_origin+np.array([0,1,0])
    icbm_origin_z=icbm_origin+np.array([0,0,1])


    center=apply_affine(icbm_to_vox, icbm_origin)

    _x=apply_affine(icbm_to_vox, icbm_origin_x)-center
    _y=apply_affine(icbm_to_vox, icbm_origin_y)-center
    _z=apply_affine(icbm_to_vox, icbm_origin_z)-center

    ix=np.argmax(np.abs(_x))
    iy=np.argmax(np.abs(_y))
    iz=np.argmax(np.abs(_z))

    center_=np.rint(center).astype(int)

    # transpose according to what we need
    img_data = np.transpose(img_data,axes=[ix, iy, iz])
    center_ = np.take(center_,[ix, iy, iz])
    sz = img_data.shape

    if _x[ix]<0:
        img_data=np.flip(img_data,axis=0)
        center_[0]=sz[0]-center_[0]
    if _y[iy]<0:
        img_data=np.flip(img_data,axis=1)
        center_[1]=sz[1]-center_[0]
    if _z[iz]<0:
        img_data=np.flip(img_data,axis=2)
        center_[2]=sz[2]-center_[0]

    slice_0 = np.take(img_data,center_[0],0)
    slice_1 = np.take(img_data,center_[1],1)
    slice_2 = np.take(img_data,center_[2],2)

    # adjust FOV, need to pad Y by 4 voxels
    # Y-Z 
    slice_0 = np.pad(slice_0[:,50:(50+193)],((4,0),(0,0)),constant_values=(0.0, 0.0),mode='constant')[0:229,:]
    # X-Z
    slice_1 = slice_1[31:(31+193),50:(50+193)]
    # X-Y
    slice_2 = np.pad(slice_2[31:(31+193),:],((0,0),(4,0)),constant_values=(0.0, 0.0),mode='constant')[:,0:229]

    _min=np.min(img_data)
    _max=np.max(img_data)

    input_images = [slice_2.T,
                    slice_0.T,
                    slice_1.T
                    ]

    input_images = [(i-_min)/(_max-_min)-0.5 for i in input_images]

    # flip, resize and crop
    for i in range(3):
        _scale = min(256.0/input_images[i].shape[0],
                     256.0/input_images[i].shape[1])
        # vertical flip and resize

        new_sz=[ math.floor(input_images[i].shape[0]*_scale),
                 math.floor(input_images[i].shape[1]*_scale) ]

        dummy=torch.full([1, 256, 256],-0.5)

        dummy[:,(256-new_sz[0])//2:(256-new_sz[0])//2+new_sz[0], 
                (256-new_sz[1])//2:(256-new_sz[1])//2+new_sz[1]] = \
            F.resize(torch.from_numpy(i).float().flip(0).unsqueeze(0),
                     size=new_sz, antialias=True)
        # crop
        input_images[i]=dummy[:,16:240,16:240]
   
    return input_images 

