import torch

def gkern3d(l=5, sigma=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sigma`
    """
    ax = torch.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = torch.exp(-0.5 * torch.square(ax) / (sigma*sigma))
    kernel = gauss[None,None,:]*gauss[None,:,None]*gauss[:,None,None]
    return kernel / torch.sum(kernel)
