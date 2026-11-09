import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from mamba_ssm import Mamba

from inpainting.training.modules.util import TimestepBlock  # 确保路径正确
import torch.utils.checkpoint as cp


os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

from torch.utils.cpp_extension import load

from .SEFN import SEFN


#================================VRWKV======================================

T_MAX = 256 * 256

# 加载 CUDA 算子
wkv_cuda = load(
    name="wkv",
    sources=["./inpainting/training/modules/CUDA/wkv_op.cpp", "./inpainting/training/modules/CUDA/wkv_cuda.cu"],
    extra_include_paths=["/usr/local/cuda/include"],
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
        self.in_proj = nn.Conv2d(n_embd, n_embd, 1)  # 1×1 Conv
        self.norm1 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.ffnv = VRWKV_ChannelMix(n_embd)
        self.dwconv = nn.Conv2d(n_embd, n_embd, 5, padding=2, groups=n_embd)  # 5×5 DW Conv
        self.out_proj = nn.Conv2d(n_embd, n_embd, 1)  # 1×1 Conv

    def forward(self, x):
        b, c, h, w = x.shape
        res = x

        x = self.in_proj(x)  # 前向通道映射
        x_ = rearrange(x, 'b c h w -> b (h w) c')
        x_ = x_ + self.att(self.norm1(x_), (h, w))
        x_ = x_ + self.ffnv(self.norm2(x_), (h, w))
        x = rearrange(x_, 'b (h w) c -> b c h w', h=h)

        x = x + self.dwconv(x)  # 加 DWConv 空间卷积增强
        x = self.out_proj(x)  # 输出映射
        x = x + res  # 最外层残差
        return x






def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def forward(self, x): return dwt_init(x)

class IWT(nn.Module):
    def forward(self, x): return iwt_init(x)

def data_transform(X): return 2 * X - 1.0
def inverse_data_transform(X): return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)





class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, t='WithBias'):
        super().__init__()
        self.body = WithBias_LayerNorm(dim) if t == 'WithBias' else BiasFree_LayerNorm(dim)
    def forward(self, x): return to_4d(self.body(to_3d(x)), x.shape[-2], x.shape[-1])

class WM(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(c, c*2, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(c*2, c, 3, 1, 1)
        )
        self.model1 = Mamba(d_model=c, d_state=32, d_conv=4, expand=2)
        # self.model2 = Mamba(d_model=c, d_state=32, d_conv=4, expand=9)
        self.smooth = nn.Conv2d(c, c, 3, 1, 1)
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c))
        y = self.model1(x).permute(0, 2, 1)
        return self.smooth(y.reshape(b, c, h, w))

# --------------------- 替换原 self.illu 为 MSGDC --------------------- #
class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2

class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x): return x + self.bias.expand_as(x)

class MSGDC(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1):
        super(MSGDC, self).__init__()
        self.move = LearnableBiasnn(in_chn)

        self.cov1 = nn.Conv2d(in_channels=in_chn, out_channels=in_chn,
                              kernel_size=kernel_size, stride=stride,
                              padding=dilation1, dilation=dilation1, bias=True)
        self.cov2 = nn.Conv2d(in_channels=in_chn, out_channels=in_chn,
                              kernel_size=kernel_size, stride=stride,
                              padding=dilation2, dilation=dilation2, bias=True)
        self.cov3 = nn.Conv2d(in_channels=in_chn, out_channels=in_chn,
                              kernel_size=kernel_size, stride=stride,
                              padding=dilation3, dilation=dilation3, bias=True)

        self.norm = nn.LayerNorm(in_chn, eps=1e-6)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)

        # Branch 1
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1, 2)
        x1 = self.act1(x1)

        # Branch 2
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1, 2)
        x2 = self.act2(x2)

        # Branch 3
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1, 2)
        x3 = self.act3(x3)

        # Combine and normalize
        x = self.norm(x1 + x2 + x3)

        # Restore shape
        return x.permute(0, 2, 1).view(B, C, H, W).contiguous()

class MSGDC_Lite(nn.Module):
    def __init__(self, in_chn):
        super(MSGDC_Lite, self).__init__()
        self.move = LearnableBiasnn(in_chn)

        self.conv1 = nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(in_chn)
        self.act1 = RPReLU(in_chn)

        self.conv2 = nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(in_chn)
        self.act2 = RPReLU(in_chn)

        self.conv3 = nn.Conv2d(in_chn, in_chn, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)

        x = self.conv1(x)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act1(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)

        x = self.conv2(x)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act2(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)

        x = self.conv3(x)
        return x





class WMB(TimestepBlock):
    def __init__(self,in_channels,emb_channels,out_channels=None,ffn_expansion_factor=2.66,bias=True,LayerNorm_type='WithBias',use_checkpoint=True,use_film=True):
        super().__init__()
        out_channels=out_channels or in_channels
        self.use_checkpoint=use_checkpoint
        self.use_proj=in_channels!=out_channels
        self.in_proj=nn.Identity() if not self.use_proj else nn.Conv2d(in_channels,out_channels,1)
        self.norm1=LayerNorm(out_channels,LayerNorm_type)


        self.low = MSGDC(out_channels)
        self.high = GlobolBlock(out_channels)
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)


        self.norm2=LayerNorm(out_channels,LayerNorm_type)
        self.ffn=SEFN(out_channels,ffn_expansion_factor,bias)
        self.use_film=use_film
        if self.use_film:
            self.time_embed=nn.Sequential(nn.SiLU(),nn.Linear(emb_channels,out_channels*2))
    def forward(self,x,emb):
        if self.use_checkpoint and self.training:
            return cp.checkpoint(self._forward_impl,x,emb)
        else:
            return self._forward_impl(x,emb)

    def _forward_impl(self, x, emb):
        x = self.in_proj(x)
        x_in = x

        x = self.norm1(x)

        # ========= 双路径（复制而非切分） =========
        x_low = x
        x_high = x

        y_low = self.low(x_low)
        y_high = self.high(x_high)

        # concat + fuse
        out = torch.cat([y_low, y_high], dim=1)
        out = self.fuse(out)

        # 主残差
        x = x + out

        # ========= FiLM 时间调制 =========
        if self.use_film:
            scale, shift = self.time_embed(emb).chunk(2, dim=1)
            while scale.ndim < x.ndim:
                scale = scale[..., None]
                shift = shift[..., None]
            x = x * (scale + 1) + shift

        # ========= FFN + 总残差 =========
        x = self.ffn(x, x_in)
        return x + x_in


# ------------------ 测试 ------------------ #

# ------------------ 测试 ------------------ #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 假设 UNet 里 time_embed_dim = 128
    block = WMB(
        in_channels=48,
        emb_channels=128,
        out_channels=48,
        use_checkpoint=False,
        use_film=True
    ).to(device)

    # 输入图像和时间嵌入
    x = torch.randn(1, 48, 192, 192).to(device)       # [B, C, H, W]
    t_emb = torch.randn(1, 128).to(device)          # [B, emb_channels]

    out = block(x, t_emb)

    print("Input size:", x.shape)
    print("Output size:", out.shape)
    total_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params/1e6:.2f} M")
# from torchsummary import summary
#
# summary(block, input_size=(48, 192, 192))