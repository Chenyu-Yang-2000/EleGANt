import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional

from models.modules.tps_transform import tps_sampler, tps_spatial_transform


def expand_area(mask:torch.Tensor, margin:int):
    '''
    mask: (C, H, W) or (N, C, H, W)
    '''
    kernel = np.zeros((margin * 2 + 1, margin * 2 + 1), dtype=np.uint8)
    kernel = cv2.circle(kernel, (margin, margin), margin, (255, 0, 0), -1)
    kernel = torch.FloatTensor((kernel > 0)).unsqueeze(0).unsqueeze(0).to(mask.device)
    ndim = mask.ndimension()
    if ndim == 3:
        mask = mask.unsqueeze(0)
    expanded_mask = torch.zeros_like(mask)
    for i in range(mask.shape[1]):
        expanded_mask[:,i:i+1,:,:] = F.conv2d(mask[:,i:i+1,:,:], kernel, padding=margin)
    if ndim == 3:
        expanded_mask = expanded_mask.squeeze(0)
    return (expanded_mask > 0).float()

def mask_blur(mask:torch.Tensor, blur_size=3, mode='smooth'):
    """Blur the edge of mask so that the compose image have smooth transition
    Args:
        mask (torch.Tensor): [C, H, W]
        blur_size (int): size of blur kernel. Defaults to 3.
        mode (str) Defaults to 'smooth'.
    Returns:
        torch.Tensor: blurred mask
    """
    #kernel = torch.ones((1, 1, blur_size * 2 + 1, blur_size * 2 + 1)).to(mask.device)
    kernel = np.zeros((blur_size * 2 + 1, blur_size * 2 + 1), dtype=np.uint8)
    kernel = cv2.circle(kernel, (blur_size, blur_size), blur_size, (255, 0, 0), -1)
    kernel = torch.FloatTensor((kernel > 0)).unsqueeze(0).unsqueeze(0).to(mask.device)
    kernel = kernel / torch.sum(kernel)
    ndim = mask.ndimension()
    if ndim == 3:
        mask = mask.unsqueeze(0)
    mask_blur = torch.zeros_like(mask)
    for i in range(mask.shape[1]):
        mask_blur[:,i:i+1,:,:] = F.conv2d(mask[:,i:i+1,:,:], kernel, padding=blur_size)
    if mode == 'valid':
        mask_blur = (mask_blur.clamp(0.5, 1) - 0.5) * 2 * mask
    if ndim == 3:
        mask_blur = mask_blur.squeeze(0)
    return mask_blur.clamp(0, 1)

def mask_blend(mask, blend_alpha, mask_bound=None, blur_size=3, blend_mode='smooth'):
    if blur_size > 0:
        mask = mask_blur(mask, blur_size, blend_mode)
    mask = mask * blend_alpha
    if mask_bound is None:
        return mask
    else:
        return mask * mask_bound


def tps_align(img_size, lms_r, lms_s, image_r, image_s=None, 
              mask_r = None, mask_s=None, sample_mode='bilinear'):
    '''
    image: (C, H, W), lms: (K, 2), mask:(1, H, W)
    '''
    lms_s = torch.flip(lms_s, dims=[1]) / (img_size - 1)
    lms_r = (torch.flip(lms_r, dims=[1]) / (img_size - 1)).unsqueeze(0)
    image_r = image_r.unsqueeze(0)
    image_trans, _ = tps_spatial_transform(img_size, img_size, lms_s, image_r, lms_r, sample_mode)
    if mask_r is not None:
        mask_r_trans, _ = tps_spatial_transform(img_size, img_size, lms_s, mask_r.unsqueeze(0), 
                                                lms_r, 'nearest')
    if image_s is not None:
        mask_compose = torch.ones((1, img_size, img_size), device=lms_r.device)
        if mask_s is not None:
            mask_compose *= mask_s
        if mask_r is not None:
            mask_compose *= mask_r_trans.squeeze(0)
        return image_s * (1 - mask_compose) + image_trans.squeeze(0) * mask_compose
    else:
        return image_trans.squeeze(0)

def tps_blend(blend_alpha, img_size, lms_r, lms_s, image_r, image_s=None, mask_r = None, mask_s=None, 
              mask_s_bound=None, blur_size=7, sample_mode='bilinear', blend_mode='smooth'):
    '''
    image: (C, H, W), lms: (K, 2), mask:(1, H, W)
    '''
    lms_s = torch.flip(lms_s, dims=[1]) / (img_size - 1)
    lms_r = (torch.flip(lms_r, dims=[1]) / (img_size - 1)).unsqueeze(0)
    image_r = image_r.unsqueeze(0)
    image_trans, _ = tps_spatial_transform(img_size, img_size, lms_s, image_r, lms_r, sample_mode)
    if mask_r is not None:
        mask_r_trans, _ = tps_spatial_transform(img_size, img_size, lms_s, mask_r.unsqueeze(0), 
                                                lms_r, 'nearest')
    if image_s is not None:
        mask_compose = torch.ones((1, img_size, img_size), device=lms_r.device)
        if mask_s is not None:
            mask_compose *= mask_s
        if mask_r is not None:
            mask_compose *= mask_r_trans.squeeze(0)
        mask_compose = mask_blend(mask_compose, blend_alpha, mask_s_bound, blur_size, blend_mode)
        return image_s * (1 - mask_compose) + image_trans.squeeze(0) * mask_compose
    else:
        return image_trans.squeeze(0)


def fine_align(img_size, lms_r, lms_s, image_r, image_s, mask_r, mask_s, margins, blend_alphas):
    '''
    image: (C, H, W), lms: (K, 2)
    mask: (C, H, W), lip, face, left eye, right eye
    margins: dictionary, blend_alphas: dictionary
    '''
    # skin align
    image_s = tps_blend(blend_alphas['skin'], img_size, lms_r[:60], lms_s[:60], image_r, image_s, 
                        mask_r[1:2], mask_s[1:2], mask_s[1:2], blur_size=8, blend_mode='valid')

    # lip align
    mask_s_lip = expand_area(mask_s[0:1], margins['lip'])
    mask_r_lip = expand_area(mask_r[0:1], margins['lip'])
    image_s = tps_blend(blend_alphas['lip'], img_size, lms_r[48:], lms_s[48:], image_r, image_s, 
                        mask_r_lip, mask_s_lip, mask_s[0:1], blur_size=3)

    # left eye align
    mask_s_eye = expand_area(mask_s[2:3], margins['eye'])
    mask_r_eye = expand_area(mask_r[2:3], margins['eye']) * mask_r[1:2]
    image_s = tps_blend(blend_alphas['eye'], img_size, 
                        torch.cat((lms_r[14:17], lms_r[22:27], lms_r[27:31], lms_r[42:48]), dim=0), 
                        torch.cat((lms_s[14:17], lms_s[22:27], lms_s[27:31], lms_s[42:48]), dim=0), 
                        image_r, image_s, mask_r_eye, mask_s_eye, mask_s[1:2], 
                        blur_size=5, sample_mode='nearest')

    # right eye align
    mask_s_eye = expand_area(mask_s[3:4], margins['eye'])
    mask_r_eye = expand_area(mask_r[3:4], margins['eye']) * mask_r[1:2]
    image_s = tps_blend(blend_alphas['eye'], img_size, 
                        torch.cat((lms_r[0:3], lms_r[17:22], lms_r[27:31], lms_r[36:42]), dim=0), 
                        torch.cat((lms_s[0:3], lms_s[17:22], lms_s[27:31], lms_s[36:42]), dim=0), 
                        image_r, image_s, mask_r_eye, mask_s_eye, mask_s[1:2], 
                        blur_size=5, sample_mode='nearest')

    return image_s


if __name__ == "__main__":
    pass