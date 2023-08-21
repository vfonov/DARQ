# /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018


import argparse
import os
import torch
import torchvision
import torchvision.transforms.functional as F
import math

#from skimage import io, transform
from minc2_simple import minc2_file 

from aqc_data import load_minc_images

def parse_options():

    parser = argparse.ArgumentParser(description='Generate QC pics for deep_qc',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("volume", type=str, 
                        help="Input minc volume ")
    parser.add_argument("output", type=str, 
                        help="Output image prefix: <prefix>_{0,1,2}.{sfx}")
    parser.add_argument("--sfx", type=str, 
                        help="Output image suffix",
                        default='.jpg')
    parser.add_argument("--quality", type=int, 
                        help="Jpeg quality",
                        default=92)
    parser.add_argument("--order", type=int, 
                        help="Resample order",
                        default=1)
    params = parser.parse_args()
    
    return params


if __name__ == '__main__':
    params = parse_options()

    slices = load_minc_images(params.volume)
    for i,_ in enumerate(slices):
        arr=((slices[i]+0.5)*255).clamp(0,255).to(torch.uint8)
        if params.sfx.endswith("jpg"):
            torchvision.io.write_jpeg(arr,f"{params.output}_{i}{params.sfx}",quality=params.quality)
        else:
            torchvision.io.write_png(arr,f"{params.output}_{i}{params.sfx}")
