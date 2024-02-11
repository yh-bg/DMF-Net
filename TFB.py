import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange


def conv3x3(in_channels, out_channels, stride=1, padding=1, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, *args, **kwargs)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances



class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.cross_attn = cross_attn

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        if not self.cross_attn:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, y=None):
        if self.shifted:
            x = self.cyclic_shift(x)
            if self.cross_attn:
                y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # print('forward-x: ', x.shape)   # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
        if not self.cross_attn:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head] * WV-2
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            qkv = (self.to_q(y), kv[0], kv[1])

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)


        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q * k / sqrt(d)

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0].long(), self.relative_indices[:, :, 1].long()]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]

class SwinModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, cross_attn):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
            ]))

    def forward(self, x, y=None):
        if y is None:
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
        else:
            x = self.patch_partition(x)
            y = self.patch_partition(y)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x, y)
                x = shifted_block(x, y)
            return x.permute(0, 3, 1, 2)


class CrossSwinTransformer(nn.Module):
    def __init__(self, n_feats=32, n_heads=4, head_dim=8, win_size=4, cross_module=['img', 'pan'], cat_feat=['img', 'pan']):
        super().__init__()
        self.n_blocks = 1
        self.cross_module = cross_module
        self.cat_feat = cat_feat

        if 'img' in self.cross_module:
            self.img_cross_pan = nn.ModuleList()
            for _ in range(self.n_blocks):
                self.img_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True))

        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(self.n_blocks):
                self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True))

    def forward(self, img, pan):

        for i in range(self.n_blocks):
            pan_cross_ms_feat = self.pan_cross_ms[i](pan, img)
            img_cross_pan_feat = self.img_cross_pan[i](img , pan)


        cat_list = []
        cat_list.append(pan_cross_ms_feat)
        cat_list.append(img_cross_pan_feat)


        output = torch.cat(cat_list, dim=1)
        return output

class CrossSwinTransformer_1(nn.Module):
    def __init__(self, n_feats=32, n_heads=4, head_dim=8, win_size=4, cross_module=['ms', 'pan'], cat_feat=['ms', 'pan']):
        super().__init__()
        self.n_blocks = 3
        self.cross_module = cross_module
        self.cat_feat = cat_feat

        if 'ms' in self.cross_module:
            self.ms_cross_pan = nn.ModuleList()
            for _ in range(self.n_blocks):
                self.ms_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True))

        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(self.n_blocks):
                self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True))

    def forward(self, ms, pan):

        for i in range(self.n_blocks):
            pan_cross_ms_feat = self.pan_cross_ms[i](pan, ms)
            ms_cross_pan_feat = self.ms_cross_pan[i](ms, pan)


        cat_list = []
        cat_list.append(pan_cross_ms_feat)
        cat_list.append(ms_cross_pan_feat)


        output = torch.cat(cat_list, dim=1)
        return output