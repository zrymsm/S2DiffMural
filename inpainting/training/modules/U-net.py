
from abc import abstractmethod
import math

import torch.fft as fft



import torch
import torch as th
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction

from inpainting.training.modules.WMA import WMB

from inpainting.training.modules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization,
    timestep_embedding,
    TimestepBlock

)
# from inpainting.training.modules.mambair_arch import VSSModule
# from inpainting.training.modules.VRWKV import VRWKVHybridBlock



def Fourier_filter(x, threshold, scale):
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device, dtype=x.dtype)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered



# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# class TimestepBlock(nn.Module):
#     """
#     Any module where forward() takes timestep embeddings as a second argument.
#     """
#
#     @abstractmethod
#     def forward(self, x, emb):
#         """
#         Apply the module to `x` given `emb` timestep embeddings.
#         """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # elif isinstance(layer, SpatialTransformer):
            #     x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)





class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True, dims)
            self.x_upd = Upsample(channels, True, dims)
        elif down:
            self.h_upd = Downsample(channels, True, dims)
            self.x_upd = Downsample(channels, True, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
            nn.SiLU(),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            h = x
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_layers(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        n_embed=None,
        b1=1.3, b2=1.4, s1=0.9, s2=0.2,  # FreeU 参数
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        self.b1, self.b2, self.s1, self.s2 = b1, b2, s1, s2

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            linear(model_channels, time_embed_dim),
            nn.GELU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Input Blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block = WMB(ch, time_embed_dim, ch, use_checkpoint=use_checkpoint)
                self.input_blocks.append(TimestepEmbedSequential(block))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = model_channels * channel_mult[level + 1]
                down = ResBlock(
                    ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                    use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True
                ) if resblock_updown else Downsample(ch, conv_resample, dims, out_channels=out_ch)
                self.input_blocks.append(TimestepEmbedSequential(down))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # Middle Blocks
        self.middle_block = TimestepEmbedSequential(
            WMB(ch, time_embed_dim, ch, use_checkpoint=use_checkpoint),
            WMB(ch, time_embed_dim, ch, use_checkpoint=use_checkpoint)
        )

        # Output Blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    WMB(ch + ich, time_embed_dim, model_channels * mult, use_checkpoint=use_checkpoint)
                ]
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    out_ch = ch
                    up = ResBlock(
                        ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                        use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True
                    ) if resblock_updown else Upsample(ch, conv_resample, dims, out_channels=out_ch)
                    layers.append(up)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(conv_nd(dims, model_channels, out_channels, 3, padding=1))

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        if isinstance(timesteps, (int, float)):
            timesteps = torch.tensor([timesteps], device=x.device)
        emb = self.time_embed(timesteps)
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            hs_ = hs.pop()

            # # --------- FreeU skip modulate + filter ----------
            # if h.shape[1] == self.model_channels * 8:
            #     hidden_mean = h.mean(1, keepdim=True)
            #     norm = (hidden_mean - hidden_mean.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) \
            #            / (hidden_mean.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - hidden_mean.min() + 1e-5)
            #     h = torch.cat([
            #         h[:, :self.model_channels * 4] * ((self.b1 - 1) * norm + 1),
            #         h[:, self.model_channels * 4:]
            #     ], dim=1)
            #     hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
            # elif h.shape[1] == self.model_channels * 4:
            #     hidden_mean = h.mean(1, keepdim=True)
            #     norm = (hidden_mean - hidden_mean.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) \
            #            / (hidden_mean.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - hidden_mean.min() + 1e-5)
            #     h = torch.cat([
            #         h[:, :self.model_channels * 2] * ((self.b2 - 1) * norm + 1),
            #         h[:, self.model_channels * 2:]
            #     ], dim=1)
            #     hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
            # # -------------------------------------------------
            #
            h = torch.cat([h, hs_], dim=1)
            h = module(h, emb, context)

        return self.out(h.type(x.dtype))
