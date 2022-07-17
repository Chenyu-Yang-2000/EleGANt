import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention_weight(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_heads=1, dropout=0.0, bias=True):
        super(MultiheadAttention_weight, self).__init__()
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = proj_dim // num_heads
        assert self.head_dim * num_heads == self.proj_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(feature_dim, proj_dim, bias=bias)
        self.k_proj = nn.Linear(feature_dim, proj_dim, bias=bias)

    def forward(self, fea_c, fea_s, mask_c, mask_s):
        '''
        fea_c: (b, d, h, w)
        mask_c: (b, c, h, w)
        '''
        bsz, dim, h, w = fea_c.shape; mask_channel = mask_c.shape[1]

        fea_c = fea_c.view(bsz, dim, h*w).transpose(1, 2) # (b, HW, d)
        fea_s = fea_s.view(bsz, dim, h*w).transpose(1, 2)
        with torch.no_grad():
            if mask_c.shape[2] != h:
                mask_c = F.interpolate(mask_c, size=(h, w)) 
                mask_s = F.interpolate(mask_s, size=(h, w)) 
            mask_c = mask_c.view(bsz, mask_channel, -1, h*w) # (b, m_c, 1, HW)
            mask_s = mask_s.view(bsz, mask_channel, -1, h*w)
            mask_attn = torch.matmul(mask_c.transpose(-2, -1), mask_s) # (b, m_c, HW, HW)
            mask_attn = torch.sum(mask_attn, dim=1, keepdim=True).clamp_(0, 1) # (b, 1, HW, HW)
            mask_sum = torch.sum(mask_attn, dim=-1, keepdim=True)
            mask_attn += (mask_sum == 0).float()
            mask_attn = mask_attn.masked_fill_(mask_attn == 0, float('-inf')).masked_fill_(mask_attn == 1, float(0.0))

        query = self.q_proj(fea_c) # (b, HW, D)
        key = self.k_proj(fea_s) # (b, HW, D)
        query = query.view(bsz, h*w, self.num_heads, self.head_dim).transpose(1, 2) # (b, h, HW, D)
        key = key.view(bsz, h*w, self.num_heads, self.head_dim).transpose(1, 2)

        weights = torch.matmul(query, key.transpose(-1, -2)) # (b, h, HW, HW)
        weights = weights * self.scaling
        weights = weights + mask_attn.detach()
        weights = self.dropout(F.softmax(weights, dim=-1))
        weights = weights * (1 - (mask_sum == 0).float().detach())
        return weights 


class MultiheadAttention_value(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_heads=1, bias=True):
        super(MultiheadAttention_value, self).__init__()
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        assert self.head_dim * num_heads == self.proj_dim, "embed_dim must be divisible by num_heads"
        
        self.v_proj = nn.Linear(feature_dim, proj_dim, bias=bias)

    def forward(self, weights, fea):
        '''
        weights: (b, h, HW. HW)
        fea: (b, d, H, W)
        '''
        bsz, dim, h, w = fea.shape
        fea = fea.view(bsz, dim, h*w).transpose(1, 2) #(b, HW, D)
        value = self.v_proj(fea)
        value = value.view(bsz, h*w, self.num_heads, self.head_dim).transpose(1, 2) #(b, h, HW, D)

        out = torch.matmul(weights, value)
        out = out.transpose(1, 2).contiguous().view(bsz, h*w, self.proj_dim) # (b, HW, D)
        out = out.transpose(1, 2).view(bsz, self.proj_dim, h, w) #(b, d, H, W)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, proj_channels, value_channels, out_channels, num_heads=1, dropout=0.0, bias=True):
        super(MultiheadAttention, self).__init__()
        self.weight = MultiheadAttention_weight(in_channels, proj_channels, num_heads, dropout, bias)
        self.value = MultiheadAttention_value(value_channels, out_channels, num_heads, bias)

    def forward(self, fea_q, fea_k, fea_v, mask_q, mask_k):
        '''
        fea: (b, d, h, w)
        mask: (b, c, h, w)
        '''
        weights = self.weight(fea_q, fea_k, mask_q, mask_k)
        return self.value(weights, fea_v)


class FeedForwardLayer(nn.Module):
    def __init__(self, feature_dim, ff_dim, dropout=0.0):
        super(FeedForwardLayer, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv2d(feature_dim, ff_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ff_dim, feature_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x)


class Attention_apply(nn.Module):
    def __init__(self, feature_dim, normalize=True):
        super(Attention_apply, self).__init__()
        self.normalize = normalize
        if normalize:
            self.norm = nn.InstanceNorm2d(feature_dim, affine=False)
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, attn_out):
        if self.normalize:
            x = self.norm(x) 
        x = x * (1 + attn_out)
        return self.conv(self.actv(x))
