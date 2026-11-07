"""Allen-Cahn problem definition, reference data, sampling and error metrics.

    u_t - 1e-4 u_xx + 5 u^3 - 5 u = 0,   x in [-1, 1], t in [0, 1]
    u(x, 0) = x^2 cos(pi x)
    u(-1, t) = u(1, t),  u_x(-1, t) = u_x(1, t)
"""
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.stats import qmc

EPS = 1e-4
DATA = Path(__file__).resolve().parents[1] / "data" / "AC.mat"


def u0(x):
    """Initial condition, works for torch tensors and numpy arrays."""
    lib = torch if isinstance(x, torch.Tensor) else np
    return x ** 2 * lib.cos(lib.pi * x)


def load_reference(path=DATA):
    """Chebfun reference solution from Raissi et al. Returns x (512,), t (201,), u (512, 201)."""
    d = scipy.io.loadmat(str(path))
    return d["x"].flatten(), d["tt"].flatten(), np.real(d["uu"])


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residual(model, x, t):
    """PDE residual at points (x, t); both tensors must require grad."""
    u = model(x, t)
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t - EPS * u_xx + 5 * u ** 3 - 5 * u


def boundary_mismatch(model, t):
    """Periodic BC mismatches (u(-1,t)-u(1,t), u_x(-1,t)-u_x(1,t))."""
    xl = torch.full_like(t, -1.0).requires_grad_(True)
    xr = torch.full_like(t, 1.0).requires_grad_(True)
    ul, ur = model(xl, t), model(xr, t)
    uxl, uxr = _grad(ul, xl), _grad(ur, xr)
    return ul - ur, uxl - uxr


class Points:
    """Training point sets. Mirrors the official AC.py: LHS collocation points,
    the full 512-point initial grid, and LHS times for the boundary."""

    def __init__(self, n_f=20000, n_b=100, seed=0, device="cpu", dtype=torch.float32):
        rng = np.random.default_rng(seed)
        lhs = qmc.LatinHypercube(d=2, seed=rng).random(n_f)
        xf = 2 * lhs[:, 0:1] - 1
        tf = lhs[:, 1:2]
        x_grid, _, _ = load_reference()
        x0 = x_grid[:, None]
        tb = qmc.LatinHypercube(d=1, seed=rng).random(n_b)
        to = dict(device=device, dtype=dtype)
        self.x_f = torch.tensor(xf, **to)
        self.t_f = torch.tensor(tf, **to)
        self.x_0 = torch.tensor(x0, **to)
        self.t_0 = torch.zeros_like(self.x_0)
        self.u_0 = u0(self.x_0)
        self.t_b = torch.tensor(tb, **to)


class Reference:
    """Reference grid for the relative L2 error."""

    def __init__(self, device="cpu", dtype=torch.float32):
        x, t, u = load_reference()
        X, T = np.meshgrid(x, t, indexing="ij")  # (512, 201)
        self.shape = X.shape
        self.x, self.t = x, t
        self.u = u
        self.X = torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype)
        self.T = torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype)

    @torch.no_grad()
    def predict(self, model):
        return model(self.X, self.T).cpu().numpy().reshape(self.shape)

    def rel_l2(self, model):
        pred = self.predict(model)
        return float(np.linalg.norm(pred - self.u) / np.linalg.norm(self.u))

    def slice_errors(self, model, times=(0.0, 0.25, 0.5, 0.75, 1.0)):
        pred = self.predict(model)
        out = {}
        for tt in times:
            j = int(np.argmin(np.abs(self.t - tt)))
            out[tt] = float(np.linalg.norm(pred[:, j] - self.u[:, j]) / np.linalg.norm(self.u[:, j]))
        return out
