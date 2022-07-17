from __future__ import absolute_import

import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

# TF32 is not enough, require FP32
# Disable automatic TF32 since Pytorch 1.7
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def grid_sample(input, grid, mode='bilinear', canvas=None):
    output = F.grid_sample(input, grid, mode=mode, align_corners=True)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid, mode='nearest', align_corners=True)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    #repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist + 1e-8)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


# compute \Delta_c^-1
def bulid_delta_inverse(target_control_points):
    '''
    target_control_points: (N, 2)
    '''
    N = target_control_points.shape[0]
    forward_kernel = torch.zeros(N + 3, N + 3).to(target_control_points.device)
    target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
    forward_kernel[:N, :N].copy_(target_control_partial_repr)
    forward_kernel[:N, -3].fill_(1)
    forward_kernel[-3, :N].fill_(1)
    forward_kernel[:N, -2:].copy_(target_control_points)
    forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
    # compute inverse matrix
    inverse_kernel = torch.inverse(forward_kernel)
    return inverse_kernel


# create target coordinate matrix
def build_target_coordinate_matrix(target_height, target_width, target_control_points):
    '''
    target_control_points: (N, 2)
    '''
    HW = target_height * target_width
    target_coordinate = list(itertools.product(range(target_height), range(target_width)))
    target_coordinate = torch.Tensor(target_coordinate).to(target_control_points.device) # HW x 2
    Y, X = target_coordinate.split(1, dim = 1)
    Y = Y / (target_height - 1)
    X = X / (target_width - 1)
    target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
    target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
    target_coordinate_repr = torch.cat([
        target_coordinate_partial_repr, 
        torch.ones((HW, 1), device=target_control_points.device), 
        target_coordinate], dim = 1)
    return target_coordinate_repr


def tps_sampler(target_height, target_width, inverse_kernel, target_coordinate_repr,
                source, source_control_points, sample_mode='bilinear'):
    '''
    inverse_kernel: \Delta_C^-1
    target_coordinate_repr: \hat{p}
    source: (B, C, H, W)
    source_control_points: (B, N, 2)
    '''
    batch_size = source.shape[0]
    Y = torch.cat([source_control_points, torch.zeros((batch_size, 3, 2), device=source.device)], dim=1)
    mapping_matrix = torch.matmul(inverse_kernel, Y)
    source_coordinate = torch.matmul(target_coordinate_repr, mapping_matrix)

    grid = source_coordinate.view(-1, target_height, target_width, 2)
    grid = torch.clamp(grid, 0, 1) # the source_control_points may be out of [0, 1].
    # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
    grid = 2.0 * grid - 1.0
    output_maps = grid_sample(source, grid, mode=sample_mode, canvas=None)
    return output_maps, source_coordinate


def tps_spatial_transform(target_height, target_width, target_control_points, 
                          source, source_control_points, sample_mode='bilinear'):
    '''
    target_control_points: (N, 2)
    source: (B, C, H, W)
    source_control_points: (B, N, 2)
    '''
    inverse_kernel = bulid_delta_inverse(target_control_points)
    target_coordinate_repr = build_target_coordinate_matrix(target_height, target_width, target_control_points)
    
    return tps_sampler(target_height, target_width, inverse_kernel, target_coordinate_repr, 
                       source, source_control_points, sample_mode)


class TPSSpatialTransformer(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSSpatialTransformer, self).__init__()
        self.target_height, self.target_width = target_height, target_width
        self.num_control_points = target_control_points.shape[0]
    
        # create padded kernel matrix
        inverse_kernel = bulid_delta_inverse(target_control_points)
    
        # create target coordinate matrix
        target_coordinate_repr = build_target_coordinate_matrix(target_height, target_width, target_control_points)
    
        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        #self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        self.register_buffer('target_control_points', target_control_points)
    
    def forward(self, source, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        
        return tps_sampler(self.target_height, self.target_width,
                           self.inverse_kernel, self.target_coordinate_repr,
                           source, source_control_points)
 