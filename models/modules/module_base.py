import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):            
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.skip = nn.Identity() if dim_in == dim_out else nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.skip(x) + self.main(x)
        return x / math.sqrt(2)


class ResidualBlock_IN(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, dim_in, dim_out, affine=False):            
        super(ResidualBlock_IN, self).__init__()
        self.main = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.skip = nn.Identity() if dim_in == dim_out else nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.skip(x) + self.main(x)
        return x / math.sqrt(2)


class ResidualBlock_Downsample(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, dim_in, dim_out, affine=False):            
        super(ResidualBlock_Downsample, self).__init__()
        self.main = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)    
        )
        if dim_in == dim_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        skip = F.interpolate(self.skip(x), scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        res = self.main(x)
        x = skip + res
        return x / math.sqrt(2)


class Downsample(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, dim_in, dim_out, affine=False):            
        super(Downsample, self).__init__()
        self.main = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)           
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock_Upsample(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, dim_in, dim_out, normalize=True, affine=False):            
        super(ResidualBlock_Upsample, self).__init__()
        if normalize:
            self.main = nn.Sequential(
                nn.InstanceNorm2d(dim_in, affine=affine),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)
            )
        else:
            self.main = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)
            )
        if dim_in == dim_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.main(x)
        x = skip + res
        return x / math.sqrt(2)


class Upsample(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, dim_in, dim_out, normalize=True, affine=False):            
        super(Upsample, self).__init__()
        if normalize:
            self.main = nn.Sequential(
                nn.InstanceNorm2d(dim_in, affine=affine),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)
            )
        else:
            self.main = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.main(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim=136, feature_size=64, max_size=None, embedding_type='l2_norm'):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_size = feature_size
        self.max_size = max_size
        assert embedding_type in ['l2_norm', 'uniform', 'sin']
        self.embedding_type = embedding_type

    @torch.no_grad()
    def forward(self, diff, mask):
        '''
        diff: (b, d, h, w)
        mask: (b, 3, h, w)
        return: (b, d, h, w)
        '''
        bsz, init_dim, init_size, _ = diff.shape
        assert self.embedding_dim >= init_dim
        diff = F.interpolate(diff, self.feature_size) # (b, d, h, w)
        mask = F.interpolate(mask, size=self.feature_size)
        mask = torch.sum(mask, dim=1, keepdim=True) # (b, 1, h, w)
        diff = diff * mask
        
        if self.embedding_type == 'l2_norm':
            norm = torch.norm(diff, dim=1, keepdim=True)
            norm = (norm == 0) + norm
            diff = diff / norm
        elif self.embedding_type == 'uniform':
            diff = diff / self.max_size
        elif self.embedding_type == 'sin':
            diff = torch.sin(diff * math.pi / (2 * self.max_size))
        
        if self.embedding_dim > init_dim:
            zero_shape = (bsz, self.embedding_dim - init_dim, self.feature_size, self.feature_size)
            zero_padding = torch.zeros(zero_shape, device=diff.device)
            diff = torch.cat((diff, zero_padding), dim=1)

        diff = diff.detach(); diff.requires_grad = False
        return diff

class MergeBlock(nn.Module):
    def __init__(self, merge_mode, feature_dim, normalize=True):
        super(MergeBlock, self).__init__()
        assert merge_mode in ['conv', 'add', 'affine']
        self.merge_mode = merge_mode
        if merge_mode == 'affine':
            self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False) if normalize else nn.Identity()
        else:
            self.norm = nn.InstanceNorm2d(feature_dim, affine=False) if normalize else nn.Identity()
        self.norm_r = nn.InstanceNorm2d(feature_dim, affine=False) if normalize else nn.Identity()
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        if merge_mode == 'conv':
            self.conv = nn.Conv2d(2 * feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, fea_s, fea_r):
        if self.merge_mode == 'conv':
            fea_s = self.norm(fea_s)
            fea_r = self.norm_r(fea_r)
            fea_s = torch.cat((fea_s, fea_r), dim=1)
        elif self.merge_mode == 'add':
            fea_s = self.norm(fea_s)
            fea_r = self.norm_r(fea_r)
            fea_s = (fea_s + fea_r) / math.sqrt(2)
        elif self.merge_mode == 'affine':
            fea_s = fea_s.permute(0, 2, 3, 1)
            fea_s = self.norm(fea_s)
            fea_s = fea_s.permute(0, 3, 1, 2)
            fea_s = fea_s * (1 + fea_r)
        return self.conv(self.actv(fea_s))