"""A small pixel-space diffusion transformer whose attention can be swapped.

Every pixel is a token (no VAE, no patchification). Tokens are put in Z-order so
LLSA's block pooling works on spatial neighbourhoods, and the model is trained with
flow matching using a rescaled noise level for larger images.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import full_attention, llsa
from .layout import zorder_permutation


@dataclass
class DiTConfig:
    size: int = 32
    channels: int = 1
    dim: int = 64
    heads: int = 4
    depth: int = 4
    attention: str = "llsa"  # "llsa" or "full"
    block: int = 4
    topk: int = 4


def noise_scale(size):
    """Noise rescaling for images above 64x64, which keeps the effective SNR comparable."""
    return size / 64 if size > 64 else 1.0


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = mx.exp(-math.log(10000) * mx.arange(half) / half)
    angles = t[:, None] * 1000 * freqs[None]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.q_norm = nn.RMSNorm(cfg.dim // cfg.heads)
        self.k_norm = nn.RMSNorm(cfg.dim // cfg.heads)
        self.out = nn.Linear(cfg.dim, cfg.dim)

    def __call__(self, x):
        B, N, D = x.shape
        H = self.cfg.heads
        q, k, v = (t.reshape(B, N, H, -1).transpose(0, 2, 1, 3) for t in mx.split(self.qkv(x), 3, axis=-1))
        q, k = self.q_norm(q), self.k_norm(k)
        if self.cfg.attention == "llsa":
            o = llsa(q, k, v, block=self.cfg.block, topk=self.cfg.topk)
        else:
            o = full_attention(q, k, v)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim, affine=False)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.dim, affine=False)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(), nn.Linear(4 * cfg.dim, cfg.dim))
        self.modulation = nn.Linear(cfg.dim, 6 * cfg.dim)
        self.modulation.weight = mx.zeros_like(self.modulation.weight)  # adaLN-zero
        self.modulation.bias = mx.zeros_like(self.modulation.bias)

    def __call__(self, x, cond):
        s1, b1, g1, s2, b2, g2 = mx.split(self.modulation(nn.silu(cond))[:, None], 6, axis=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + s1) + b1)
        return x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)


class PixelDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n = cfg.size * cfg.size
        perm = zorder_permutation(cfg.size)
        self._perm = mx.array(perm)
        self._inv = mx.array(np.argsort(perm))
        self.embed = nn.Linear(cfg.channels, cfg.dim)
        self.pos = mx.random.normal((n, cfg.dim)) * 0.02
        self.time = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.blocks = [DiTBlock(cfg) for _ in range(cfg.depth)]
        self.final_norm = nn.LayerNorm(cfg.dim, affine=False)
        self.final_mod = nn.Linear(cfg.dim, 2 * cfg.dim)
        self.final_mod.weight = mx.zeros_like(self.final_mod.weight)
        self.final_mod.bias = mx.zeros_like(self.final_mod.bias)
        self.head = nn.Linear(cfg.dim, cfg.channels)

    def __call__(self, x, t):
        """x: (B, size, size, channels); t: (B,) in [0, 1]. Returns the predicted velocity."""
        B = x.shape[0]
        tokens = x.reshape(B, -1, self.cfg.channels)[:, self._perm]
        h = self.embed(tokens) + self.pos
        cond = self.time(timestep_embedding(t, self.cfg.dim))
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = mx.split(self.final_mod(nn.silu(cond))[:, None], 2, axis=-1)
        out = self.head(self.final_norm(h) * (1 + scale) + shift)
        return out[:, self._inv].reshape(x.shape)


def flow_loss(model, x0, key=None):
    """Flow matching with x_t = (1 - t) x0 + s t eps, whose velocity is s eps - x0."""
    s = noise_scale(model.cfg.size)
    t = mx.random.uniform(shape=(x0.shape[0],), key=key)
    eps = mx.random.normal(x0.shape)
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + s * tb * eps
    return ((model(xt, t) - (s * eps - x0)) ** 2).mean()


def sample(model, n, steps=20):
    s = noise_scale(model.cfg.size)
    shape = (n, model.cfg.size, model.cfg.size, model.cfg.channels)
    x = s * mx.random.normal(shape)
    for i in range(steps):
        t = 1 - i / steps
        x = x - (1 / steps) * model(x, mx.full((n,), t))
        mx.eval(x)
    return x
