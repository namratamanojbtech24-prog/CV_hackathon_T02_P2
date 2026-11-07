"""Compact RAW-domain U-Net used by both the learned and the hybrid ISP.

Input  : packed Bayer planes (4) + per-plane noise-level map (4)  at HALF resolution   -> 8 channels
Context: 8-number global descriptor (channel means / highlights / noise / gain) that is FiLM-injected
         at the bottleneck. It gives the net scene-level knowledge (illuminant, exposure) that a
         purely local convolutional net cannot see, and — because it is computed once on the whole
         frame — keeps tiled inference seam-free.
Output : 12 channels at half resolution, pixel-shuffled to a 3-channel residual at FULL resolution.

There is deliberately no normalisation layer: batch-norm statistics shift with the noise level, and
without it quantisation/fusion is trivial and tiles are processed identically.
"""
import torch
from torch import nn
import torch.nn.functional as F

CTX_DIM = 8
IN_CH = 8


def conv(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True))


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.a, self.b = conv(ci, co), conv(co, co)

    def forward(self, x):
        return self.b(self.a(x))


class UpBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.reduce = conv(ci, co)            # applied after nearest-neighbour upsampling
        self.block = Block(2 * co, co)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.block(torch.cat((self.reduce(x), skip), 1))


class ISPUNet(nn.Module):
    """Residual RAW -> RGB U-Net. `forward(x, ctx)` returns a (B,3,2H,2W) residual."""

    def __init__(self, base=16, in_ch=IN_CH, ctx_dim=CTX_DIM, zero_init=True):
        super().__init__()
        b = base
        self.base, self.in_ch, self.ctx_dim = b, in_ch, ctx_dim
        self.e1, self.e2, self.e3 = Block(in_ch, b), Block(b, 2 * b), Block(2 * b, 4 * b)
        self.down1 = nn.Conv2d(b, b, 2, stride=2)
        self.down2 = nn.Conv2d(2 * b, 2 * b, 2, stride=2)
        self.down3 = nn.Conv2d(4 * b, 4 * b, 2, stride=2)
        self.mid = Block(4 * b, 8 * b)
        self.film = nn.Sequential(nn.Linear(ctx_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 8 * b + b))
        self.u3, self.u2, self.u1 = UpBlock(8 * b, 4 * b), UpBlock(4 * b, 2 * b), UpBlock(2 * b, b)
        self.head = nn.Conv2d(b, 12, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        for m in self.modules():           # He init: PyTorch's default shrinks activations ~0.6x per layer, which stalls a
            if isinstance(m, (nn.Conv2d, nn.Linear)):   # normalisation-free net of this depth
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        if zero_init:                      # start as the identity correction (residual = 0)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x, ctx):
        a = self.e1(x)
        b = self.e2(self.down1(a))
        c = self.e3(self.down2(b))
        d = self.mid(self.down3(c))
        f = self.film(ctx)
        d = d + f[:, : 8 * self.base, None, None]
        y = self.u3(d, c)
        y = self.u2(y, b)
        y = self.u1(y, a)
        y = y + f[:, 8 * self.base:, None, None]
        return self.shuffle(self.head(y))


# --------------------------------------------------------------------------- utilities
def model_info(model, dtype_bytes=4):
    n = sum(p.numel() for p in model.parameters())
    return {"parameters": n, "size_mb": n * dtype_bytes / 1e6,
            "size_mb_fp16": n * 2 / 1e6, "size_mb_int8": n * 1 / 1e6}


def count_macs(model, packed_hw=(64, 64), device="cpu"):
    """Multiply-accumulates for one forward pass at the given packed (half-res) size."""
    macs = [0]

    def conv_hook(m, i, o):
        macs[0] += o.numel() // o.shape[0] * m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    x = torch.zeros(1, model.in_ch, *packed_hw, device=device)
    c = torch.zeros(1, model.ctx_dim, device=device)
    with torch.no_grad():
        model.eval()(x, c)
    for h in hooks:
        h.remove()
    return macs[0]
