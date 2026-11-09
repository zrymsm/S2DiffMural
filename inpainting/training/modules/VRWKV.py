import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse  # pip install pytorch_wavelets
from inpainting.training.modules.util import TimestepBlock
import math

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

from torch.utils.cpp_extension import load

from inpainting.training.modules.util import TimestepBlock  # 确保路径正确

import torch.utils.checkpoint as cp

T_MAX = 256 * 256

# 加载 CUDA 算子
wkv_cuda = load(
    name="wkv",
    sources=["/home/tomry/code/work2_mural_inpainting/1Dunhuang_exp/MuralViL-V4-WEFFN（最强）/MuralViL-main/src/CUDA/wkv_op.cpp", "/home/tomry/code/work2_mural_inpainting/1Dunhuang_exp/MuralViL-V4-WEFFN（最强）/MuralViL-main/src/CUDA/wkv_cuda.cu"],
    extra_include_paths=["/usr/local/cuda/include"],  # ✅ 添加这一行！
    verbose=True,
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}']
)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w.float().contiguous(), u.float().contiguous(), k.float().contiguous(), v.float().contiguous(), y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()

        wkv_cuda.backward(B, T, C,
                           w.float().contiguous(), u.float().contiguous(),
                           k.float().contiguous(), v.float().contiguous(),
                           gy.float().contiguous(), gw, gu, gk, gv)

        return None, None, None, gw.sum(dim=0), gu.sum(dim=0), gk, gv

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())






class WEFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, wave='haar'):
        super(WEFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=2, groups=hidden_features * 2,
            bias=bias, dilation=2
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 小波变换模块
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)  # 一层DWT
        self.idwt = DWTInverse(mode='zero', wave=wave)  # 对应的IDWT

        # 高频通道调制参数：shape = [B, C, 3, H, W]，三个方向（LH, HL, HH）
        self.weight_h = nn.Parameter(torch.ones((1, hidden_features * 2, 3, 1, 1)))
        self.bias_h = nn.Parameter(torch.zeros((1, hidden_features * 2, 3, 1, 1)))

    def forward(self, x):
        res = x  # 添加残差连接

        x = self.project_in(x)
        x = self.dwconv(x)

        yl, yh = self.dwt(x)  # yl: approximation (低频), yh: high frequency list with shape [B, C, 3, H, W]
        yh_modulated = self.weight_h * yh[0] + self.bias_h
        x = self.idwt((yl, [yh_modulated]))

        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.project_out(x)
        return x + res

class OmniShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)

    def forward(self, x):
        return self.conv5x5(x)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, recurrence=2):
        super().__init__()
        self.recurrence = recurrence
        self.omni_shift = OmniShift(dim=n_embd)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.randn((recurrence, n_embd)))
        self.spatial_first = nn.Parameter(torch.randn((recurrence, n_embd)))

    def forward(self, x, resolution):
        B, T, C = x.size()
        h, w = resolution
        x = rearrange(self.omni_shift(rearrange(x, 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')
        k, v, r = self.key(x), self.value(x), self.receptance(x)
        sr = torch.sigmoid(r)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h)
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h)

        x = sr * v
        return self.output(x)


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        hidden_sz = int(hidden_rate * n_embd)
        self.omni_shift = OmniShift(dim=n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(self.omni_shift(rearrange(x, 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')
        k = torch.square(F.relu(self.key(x)))
        x = torch.sigmoid(self.receptance(x)) * self.value(k)
        return x

#全局建模块
class GlobolBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd)
        self.ffnv = VRWKV_ChannelMix(n_embd)

    def forward(self, x):
        b, c, h, w = x.shape
        res = (h, w)
        x_ = rearrange(x, 'b c h w -> b (h w) c')
        x_ = x_ + self.att(self.ln1(x_), res)
        x_ = x_ + self.ffnv(self.ln2(x_), res)
        x = rearrange(x_, 'b (h w) c -> b c h w', h=h)
        return x


#局部建模块
class LocalConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, bias=False)  # 空洞卷积
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.conv1(x)
        x = self.ln1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.ln2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.act2(x)
        x = self.conv3(x)
        return x



# class VRWKVBlock(TimestepBlock):
#     def __init__(self, in_channels, emb_channels, out_channels=None, dropout=0.0, use_checkpoint=True):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         self.in_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
#         self.block = GlobolBlock(n_embd=out_channels)
#         self.time_mlp = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_channels, out_channels)
#         )
#         self.use_checkpoint = use_checkpoint
#
#     def forward(self, x, emb):
#         x = self.in_proj(x)
#         emb_out = self.time_mlp(emb)  # [B, C]
#         while len(emb_out.shape) < len(x.shape):
#             emb_out = emb_out[..., None]
#         x = x + emb_out
#
#         if self.use_checkpoint and self.training:
#             def custom_forward(inp):
#                 return self.block(inp)
#             x = cp.checkpoint(custom_forward, x)
#         else:
#             x = self.block(x)
#
#         return x
class VRWKVHybridBlock(TimestepBlock):
    def __init__(self, in_channels, emb_channels, out_channels=None, dropout=0.1, use_checkpoint=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.use_proj = in_channels != out_channels
        self.in_proj = nn.Identity() if not self.use_proj else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.use_checkpoint = use_checkpoint

        self.local1 = LocalConvBlock(out_channels)
        self.global1 = GlobolBlock(out_channels)
        self.fuse1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # 改这里 ��：和UNet的time_embed定义一致
        # self.time_embed = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(emb_channels, out_channels))

        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels * 2)  # 改为 2×输出
        )

        self.local2 = LocalConvBlock(out_channels)
        self.global2 = GlobolBlock(out_channels)
        self.fuse2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        self.weffn = WEFFN(out_channels)

    def forward(self, x, emb):
        if self.use_checkpoint and self.training:
            return cp.checkpoint(self._forward_impl, x, emb)
        else:
            return self._forward_impl(x, emb)

    def _forward_impl(self, x, emb):


        x = self.in_proj(x)
        x_in = x  # 用于残差

        x_local1 = self.local1(x)
        x_global1 = self.global1(x)
        x = self.fuse1(torch.cat([x_local1, x_global1], dim=1))
        x = x + x_in  # 残差连接1

        x = self.dropout(x)

        # ✅ FiLM-style 时间调制
        emb_out = self.time_embed(emb)  # [B, 2C]
        scale, shift = emb_out.chunk(2, dim=1)
        while len(scale.shape) < len(x.shape):
            scale = scale[..., None]
            shift = shift[..., None]
        x = x * (scale + 1) + shift



        x_local2 = self.local2(x)
        x_global2 = self.global2(x)
        x = self.fuse2(torch.cat([x_local2, x_global2], dim=1))



        x = self.weffn(x)


        return x + x_in  # 总残差


if __name__ == "__main__":
    x = torch.zeros((1, 48, 192, 192)).cuda()
    timestep = torch.randint(0, 1000, (1,))  # 假设有1个batch
    emb = torch.randn(1, 48).cuda()  # 对应的时间步embedding输入

    model = VRWKVHybridBlock(in_channels=48, emb_channels=48).cuda()
    y = model(x, emb)
    print(y.shape, "Params:", sum(p.numel() for p in model.parameters()))
