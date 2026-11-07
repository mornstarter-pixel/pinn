"""MLP with optional exact (hard) enforcement of the periodic BC and the IC."""
import math

import torch
import torch.nn as nn

from .pde import u0


class PINN(nn.Module):
    """u(x,t) = N(features(x,t)); features = (x,t) or Fourier embedding of x.

    periodic : replace x by (cos k pi x, sin k pi x), k=1..n_fourier. Any network
               output is then C^inf-periodic on [-1, 1]: the BC loss is dropped.
    hard_ic  : u = u0(x) + s(t) N(x,t) with s(0) = 0, so u(x,0) = u0(x) exactly
               and the IC loss is dropped.  s(t) = t ('linear') or 1-exp(-t).
    """

    def __init__(self, hidden=(128, 128, 128, 128), periodic=False, n_fourier=1,
                 hard_ic=False, ic_scale="linear"):
        super().__init__()
        self.periodic, self.n_fourier, self.hard_ic, self.ic_scale = periodic, n_fourier, hard_ic, ic_scale
        in_dim = 1 + 2 * n_fourier if periodic else 2
        dims = [in_dim, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(lin.weight)   # Keras glorot_normal
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)

    def features(self, x, t):
        if not self.periodic:
            return torch.cat([x, t], dim=1)
        feats = [t]
        for k in range(1, self.n_fourier + 1):
            feats += [torch.cos(k * math.pi * x), torch.sin(k * math.pi * x)]
        return torch.cat(feats, dim=1)

    def forward(self, x, t):
        h = self.features(x, t)
        for lin in self.layers[:-1]:
            h = torch.tanh(lin(h))
        n = self.layers[-1](h)
        if not self.hard_ic:
            return n
        s = t if self.ic_scale == "linear" else 1 - torch.exp(-t)
        return u0(x) + s * n

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
