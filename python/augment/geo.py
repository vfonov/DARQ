import torch as T
import torch.nn.functional as F


"""
create transformation matrix from voxel space to pytorch space
"""
def create_v2p_matrix(shape,device = None):
    v2p = T.diag( T.tensor([2/shape[2],   2/shape[1],   2/shape[0], 1],device=device))
    v2p[0:3,3] = T.tensor([  1/shape[2]-1, 1/shape[1]-1, 1/shape[0]-1  ]) # adjust for half a voxel shift
    return v2p

"""
create rotation matrix based on euler angles
"""
def create_rotation_matrix(rot, device = None, dtype=T.float32):
    sin=lambda th: T.sin(T.as_tensor(th, dtype=dtype, device=device))
    cos=lambda th: T.cos(T.as_tensor(th, dtype=dtype, device=device))
    # rotate around x
    affine_x = T.eye(4, device=device)
    sin_, cos_ = sin(rot[0]), cos(rot[0])
    affine_x[1, 1], affine_x[1, 2] = cos_, -sin_
    affine_x[2, 1], affine_x[2, 2] = sin_, cos_

    # rotate around y
    affine_y = T.eye(4, device=device)
    sin_, cos_ = sin(rot[1]), cos(rot[1])
    affine_y[0, 0], affine_y[0, 2] = cos_, sin_
    affine_y[2, 0], affine_y[2, 2] = -sin_, cos_

    # rotate around z
    affine_z = T.eye(4, device=device)
    sin_, cos_ = sin(rot[2]), cos(rot[2])
    affine_z[0, 0], affine_z[0, 1] = cos_, -sin_
    affine_z[1, 0], affine_z[1, 1] = sin_, cos_
    return  affine_x @ affine_y @ affine_z

"""
create scale matrix based on scale factors
"""
def create_scale_matrix(scale, device = None):
    return T.diag( T.tensor([*scale, 1.0],device=device))

"""
create translation matrix based on shift factors
"""
def create_translation_matrix(shift, device = None):
    affine=T.eye(4,device=device)
    affine[0:3,3] = shift
    return affine

"""
create shear matrix based on shear factors
"""
def create_shear_matrix(shear,device = None):
    affine=T.eye(4,device=device)
    affine[0, 1], affine[0, 2] = shear[0], shear[1]
    affine[1, 0], affine[1, 2] = shear[2], shear[3]
    affine[2, 0], affine[2, 1] = shear[4], shear[5]
    return affine

"""
create a full affine transformations from set of parameters
"""
def create_transform(rot, scale, shift, shear,device = None):
    Mrot=create_rotation_matrix(rot,device=device)
    Mscale=create_scale_matrix(scale,device=device)
    Mshear=create_shear_matrix(shear,device=device)
    Mtrans=create_translation_matrix(shift,device=device)
    return Mtrans @ Mshear @ Mscale @ Mrot

"""
create random rotations matrix
"""
def random_rotations(mrot, device=None):
    return create_rotation_matrix(
            T.rand(3,device=device)*2*mrot-mrot,
            device=device)

"""
create random scaling matrix
"""
def random_scales(mscale, device=None):
    return create_scale_matrix(
            T.rand(3,device=device)*2*mscale-mscale+1.0,
            device=device)
"""
create random translation matrix
"""
def random_translations(mshift,device=None):
    return create_translation_matrix(
            T.rand(3,device=device)*2*mshift-mshift,
            device=device)

"""
create random shear matrix
"""
def random_shears(mshear,device=None):
    return create_shear_matrix(
            T.rand(6,device=device)*2*mshear-mshear,
            device=device)

"""
convert affine transform matrix into pytorch style grid
"""
def affine2grid(affine, shape, dtype=T.float, device=None):
    return F.affine_grid(affine[0:3, 0:4].unsqueeze(0).to(device),
                         shape, align_corners=False ).to(dtype)


"""
Convert pytorch style grid into flow field (displacement field)
"""
def grid_to_flow(in_grid, shape=None):
    in_grid = in_grid[...,[2, 1, 0]]
    # unscale
    if shape is None:
        shape = in_grid.shape[1:4]

    vectors = [T.arange(0, s, dtype=in_grid.dtype, device=in_grid.device) + 0.5 for s in shape] 
    grids   =  T.meshgrid(vectors, indexing='ij')

    grid = T.stack(grids)
    grid = T.unsqueeze(grid, 0)

    new_locs = in_grid.permute(0, 4, 1, 2, 3)

    for i in range(3):
        new_locs[:, i, ...] = (new_locs[:, i, ...] + 1.0) * shape[i]/2.0

    return (new_locs - grid)
