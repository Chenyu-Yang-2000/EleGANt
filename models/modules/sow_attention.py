import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    def __init__(self, window_size, in_channels, proj_channels, value_channels, out_channels, 
                 num_heads=1, dropout=0.0, bias=True, weighted_output=True):
        super(WindowAttention, self).__init__()
        assert window_size % 2 == 0
        self.window_size = window_size
        self.weighted_output = weighted_output
        window_weight = self.generate_window_weight()
        self.register_buffer('window_weight', window_weight)

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        head_dim = proj_channels // num_heads
        assert head_dim * num_heads == self.proj_channels, "embed_dim must be divisible by num_heads"
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=bias)

        self.value_channels = value_channels
        self.out_channels = out_channels
        assert out_channels // num_heads * num_heads == self.out_channels
        self.v_proj = nn.Conv2d(value_channels, out_channels, kernel_size=1, bias=bias)

    @torch.no_grad()
    def generate_window_weight(self):
        yc = torch.arange(self.window_size // 2).unsqueeze(1).repeat(1, self.window_size // 2)
        xc = torch.arange(self.window_size // 2).unsqueeze(0).repeat(self.window_size // 2, 1)
        window_weight = xc * yc / (self.window_size // 2 - 1) ** 2
        window_weight = torch.cat((window_weight, torch.flip(window_weight, dims=[0])), dim=0)
        window_weight = torch.cat((window_weight, torch.flip(window_weight, dims=[1])), dim=1)
        return window_weight.view(-1)   

    def make_window(self, x: torch.Tensor):
        """
        input: (B, C, H, W)
        output: (B, h, H/S, W/S, S*S, C/h)
        """
        bsz, dim, h, w = x.shape
        x = x.view(bsz, self.num_heads, dim // self.num_heads, h // self.window_size, self.window_size, 
                   w // self.window_size, self.window_size)
        x = x.transpose(4, 5).contiguous().view(bsz, self.num_heads, dim // self.num_heads, 
                                                h // self.window_size, w // self.window_size, self.window_size**2)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def demake_window(self, x: torch.Tensor):
        """
        input: (B, h, H/S, W/S, S*S, C/h)
        output: (B, C, H, W)
        """
        bsz, _, h_s, w_s, _, dim_h = x.shape
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()
        #print(x.shape)
        x = x.view(bsz, dim_h * self.num_heads, h_s, w_s, self.window_size, self.window_size)
        #print(x.shape)
        x = x.transpose(3, 4).contiguous().view(bsz, dim_h * self.num_heads, 
                                                h_s * self.window_size, w_s * self.window_size)
        #print(x.shape)
        return x

    @torch.no_grad()
    def make_mask_window(self, mask: torch.Tensor):
        """
        input: (B, C, H, W)
        output: (B, 1, H/S, W/S, S*S, C)
        """
        bsz, mask_channel, h, w = mask.shape
        mask = mask.view(bsz, 1, mask_channel, h // self.window_size, self.window_size, 
                         w // self.window_size, self.window_size)
        mask = mask.transpose(4, 5).contiguous().view(bsz, 1, mask_channel, 
                                                      h // self.window_size, w // self.window_size, self.window_size**2)
        mask = mask.permute(0, 1, 3, 4, 5, 2)
        return mask
    
    def forward(self, fea_q, fea_k, fea_v, mask_q=None, mask_k=None):
        '''
        fea: (b, d, h, w)
        mask: (b, c, h, w)
        '''
        query = self.q_proj(fea_q) # (B, D, H, W)
        key = self.k_proj(fea_k)
        value = self.v_proj(fea_v)
        query = self.make_window(query) # (B, h, H/S, W/S, S*S, D/h)
        key = self.make_window(key)
        value = self.make_window(value)
        
        weights = torch.matmul(query, key.transpose(-1, -2)) # (B, h, H/S, W/S, S*S, S*S)
        weights = weights * self.scaling
        if mask_q is not None and mask_k is not None:
            mask_q = self.make_mask_window(mask_q) # (B, 1, H/S, W/S, S*S, C)
            mask_k = self.make_mask_window(mask_k)
            with torch.no_grad():
                mask_attn = torch.matmul(mask_q, mask_k.transpose(-1, -2))
                mask_sum = torch.sum(mask_attn, dim=-1, keepdim=True)
                mask_attn += (mask_sum == 0).float()
                mask_attn = mask_attn.masked_fill_(mask_attn == 0, float('-inf')).masked_fill_(mask_attn == 1, float(0.0))
            weights += mask_attn        

        weights = self.dropout(F.softmax(weights, dim=-1))
        if mask_q is not None and mask_k is not None:
            weights = weights * (1 - (mask_sum == 0).float().detach())

        out = torch.matmul(weights, value) # (B, h, H/S, W/S, S*S, D/h)
        if self.weighted_output:
            window_weight = self.window_weight.view(1, 1, 1, 1, self.window_size ** 2, 1)
            out = out * window_weight
        out = self.demake_window(out) #(B, D, H, W)
        return out

class SowAttention(nn.Module):
    def __init__(self, window_size, in_channels, proj_channels, value_channels, out_channels, 
                 num_heads=1, dropout=0.0, bias=True):
        super(SowAttention, self).__init__()
        assert window_size % 2 == 0
        self.window_size = window_size
        self.pad = nn.ZeroPad2d(window_size // 2)
        self.window_attention = WindowAttention(window_size, in_channels, proj_channels, value_channels,
                                            out_channels, num_heads, dropout, bias)

    def forward(self, fea_q, fea_k, fea_v, mask_q=None, mask_k=None):
        '''
        fea: (b, d, h, w)
        mask: (b, c, h, w)
        '''
        out_0 = self.window_attention(fea_q, fea_k, fea_v, mask_q, mask_k)
        
        fea_q = self.pad(fea_q)
        fea_k = self.pad(fea_k)
        fea_v = self.pad(fea_v)
        if mask_q is not None and mask_k is not None:
            mask_q = self.pad(mask_q)
            mask_k = self.pad(mask_k)
        else:
            mask_q = None; mask_k = None
        
        out_1 = self.window_attention(fea_q, fea_k, fea_v, mask_q, mask_k)
        out_1 = out_1[:, :, self.window_size//2:-self.window_size//2, self.window_size//2:-self.window_size//2]
        
        if mask_q is not None and mask_k is not None:
            out_2 = self.window_attention(
                fea_q[:, :, :, self.window_size//2:-self.window_size//2],
                fea_k[:, :, :, self.window_size//2:-self.window_size//2],
                fea_v[:, :, :, self.window_size//2:-self.window_size//2],
                mask_q[:, :, :, self.window_size//2:-self.window_size//2],
                mask_k[:, :, :, self.window_size//2:-self.window_size//2]
            )
        else:
            out_2 = self.window_attention(
                fea_q[:, :, :, self.window_size//2:-self.window_size//2],
                fea_k[:, :, :, self.window_size//2:-self.window_size//2],
                fea_v[:, :, :, self.window_size//2:-self.window_size//2],
            )
        out_2 = out_2[:, :, self.window_size//2:-self.window_size//2, :]

        if mask_q is not None and mask_k is not None:
            out_3 = self.window_attention(
                fea_q[:, :, self.window_size//2:-self.window_size//2, :],
                fea_k[:, :, self.window_size//2:-self.window_size//2, :],
                fea_v[:, :, self.window_size//2:-self.window_size//2, :],
                mask_q[:, :, self.window_size//2:-self.window_size//2, :],
                mask_k[:, :, self.window_size//2:-self.window_size//2, :]
            )
        else:
            out_3 = self.window_attention(
                fea_q[:, :, self.window_size//2:-self.window_size//2, :],
                fea_k[:, :, self.window_size//2:-self.window_size//2, :],
                fea_v[:, :, self.window_size//2:-self.window_size//2, :],
            )
        out_3 = out_3[:, :, :, self.window_size//2:-self.window_size//2]

        out = out_0 + out_1 + out_2 + out_3
        return out

class StridedwindowAttention(nn.Module):
    def __init__(self, stride, in_channels, proj_channels, value_channels, out_channels, 
                 num_heads=1, dropout=0.0, bias=True):
        super(StridedwindowAttention, self).__init__()
        self.stride = stride
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        head_dim = proj_channels // num_heads
        assert head_dim * num_heads == self.proj_channels, "embed_dim must be divisible by num_heads"
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=bias)

        self.value_channels = value_channels
        self.out_channels = out_channels
        assert out_channels // num_heads * num_heads == self.out_channels
        self.v_proj = nn.Conv2d(value_channels, out_channels, kernel_size=1, bias=bias)

    def make_window(self, x: torch.Tensor):
        """
        input: (B, C, H, W)
        output: (B, h, S(h), S(w), H/S * W/S, C/h)
        """
        bsz, dim, h, w = x.shape
        assert h % self.stride == 0 and w % self.stride == 0
        
        x = x.view(bsz, self.num_heads, dim // self.num_heads, h // self.stride, self.stride, 
                   w // self.stride, self.stride) # (B, h, C/h, H/S, S(h), W/S, S(w))
        x = x.permute(0, 1, 4, 6, 3, 5, 2).contiguous() # (B, h, S(h), S(w), H/S, W/S, C/h)
        x = x.view(bsz, self.num_heads, self.stride, self.stride,  
                   h // self.stride * w // self.stride, dim // self.num_heads)
        return x

    def demake_window(self, x: torch.Tensor, h, w):
        """
        input: (B, h, S(h), S(w), H/S * W/S, C/h)
        output: (B, C, H, W)
        """
        bsz, _, _, _, _, dim_h = x.shape
        x = x.view(bsz, self.num_heads, self.stride, self.stride,  
                   h // self.stride, w // self.stride, dim_h) # (B, h, S(h), S(w), H/S, W/S, C/h)
        x = x.permute(0, 1, 6, 4, 2, 5, 3).contiguous() # (B, h, C/h, H/S, S(h), W/S, S(w))
        x = x.view(bsz, dim_h * self.num_heads, h, w)
        return x

    @torch.no_grad()
    def make_mask_window(self, mask: torch.Tensor):
        """
        input: (B, C, H, W)
        output: (B, 1, S(h), S(w), H/S * W/S, C)
        """
        bsz, mask_channel, h, w = mask.shape
        assert h % self.stride == 0 and w % self.stride == 0

        mask = mask.view(bsz, 1, mask_channel, h // self.stride, self.stride, w // self.stride, self.stride)
        mask = mask.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
        mask = mask.view(bsz, 1, self.stride, self.stride, h // self.stride * w // self.stride, mask_channel)
        return mask
    
    def forward(self, fea_q, fea_k, fea_v, mask_q=None, mask_k=None):
        '''
        fea: (b, d, h, w)
        mask: (b, c, h, w)
        '''
        bsz, _, h, w = fea_q.shape
        
        query = self.q_proj(fea_q) # (B, D, H, W)
        key = self.k_proj(fea_k)
        value = self.v_proj(fea_v)
        query = self.make_window(query) # (B, h, S(h), S(w), H/S * W/S, C/h)
        key = self.make_window(key)
        value = self.make_window(value)
        
        weights = torch.matmul(query, key.transpose(-1, -2)) # (B, h, S(h), S(w), H/S * W/S, H/S * W/S)
        weights = weights * self.scaling
        if mask_q is not None and mask_k is not None:
            mask_q = self.make_mask_window(mask_q) # (B, 1, S(h), S(w), H/S * W/S, C)
            mask_k = self.make_mask_window(mask_k)
            with torch.no_grad():
                mask_attn = torch.matmul(mask_q, mask_k.transpose(-1, -2))
                mask_sum = torch.sum(mask_attn, dim=-1, keepdim=True)
                mask_attn += (mask_sum == 0).float()
                mask_attn = mask_attn.masked_fill_(mask_attn == 0, float('-inf')).masked_fill_(mask_attn == 1, float(0.0))
            weights += mask_attn        

        weights = self.dropout(F.softmax(weights, dim=-1))
        if mask_q is not None and mask_k is not None:
            weights = weights * (1 - (mask_sum == 0).float().detach())

        out = torch.matmul(weights, value) # (B, h, S(h), S(w), H/S * W/S, D/h)
        out = self.demake_window(out, h, w) #(B, D, H, W)
        return out
    