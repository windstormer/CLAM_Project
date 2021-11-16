import denseCRF
import os
import numpy as np
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)
from utils import *

def gen_seg_mask(img, cam, img_name, result_path):

    threshold = (cam.max()+cam.min())/2
    if threshold < 0.1:
        first_seg = np.zeros_like(cam)
        final_seg = first_seg
    else:
        first_seg = np.where(cam>threshold, 1, 0)
        final_seg = DCRF(img, first_seg)
        # final_seg = morphGAC(img, first_seg)
        # final_seg = first_seg

    # threshold = 0.3
    # first_seg = np.where(cam>threshold, 1, 0)
    # final_seg = first_seg
    # final_seg = morphGAC(img, first_seg)
    # final_seg = DCRF(img, first_seg)

    # final_seg = first_seg
    print_hist(result_path, cam, threshold, img_name)
    

    return first_seg, final_seg

def morphGAC(img, first_seg):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gimage = inverse_gaussian_gradient(gray)
    final_seg = morphological_geodesic_active_contour(gimage, 300,
                                        init_level_set=first_seg,
                                        smoothing=2, balloon=-1)
    
    return final_seg

def DCRF(img, first_seg):
    img = np.asarray(img)
    img = (img*255).astype(np.uint8)

    first_seg = first_seg.astype(np.float32)
    prob = np.repeat(first_seg[..., np.newaxis], 3, axis=2)
    prob = prob[:, :, :2]
    prob[:, :, 0] = 1.0 - prob[:, :, 0]
    w1    = 10.0  # weight of bilateral term
    alpha = 10    # spatial std
    beta  = 13    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 50   # iteration

    # w1    = 4  # weight of bilateral term
    # alpha = 67    # spatial std
    # beta  = 3    # rgb  std
    # w2    = 3.0   # weight of spatial term
    # gamma = 1     # spatial std
    # it    = 10   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    final_seg = denseCRF.densecrf(img, prob, param)
    # print(final_seg.shape)
    return final_seg

