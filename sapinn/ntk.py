"""Empirical neural tangent kernel (NTK) analysis of the PINN loss terms.

For a vector of loss residuals f(theta) in R^N (PDE residual, IC mismatch,
BC mismatch) the empirical NTK is K = J J^T with J = df/dtheta.  Under
gradient flow on L = 1/2 ||f||^2 the residuals obey df/dt = -K f (Wang, Yu &
Perdikaris, JCP 2022), so the eigenvalues of K set the convergence rate of each
mode.  With self-adaptive weights lambda the trained vector is Lambda f, whose
kernel is Lambda K Lambda; with a mean-squared loss over N_i points the block i
is further scaled by 1/N_i.  Hard constraints simply delete the IC/BC blocks.
"""
import numpy as np
import torch

from .pde import boundary_mismatch, residual


def _flat_jacobian(f, params):
    """J (N, P) of the vector f w.r.t. the parameter list."""
    N = f.numel()
    f = f.reshape(-1)
    try:
        eye = torch.eye(N, device=f.device, dtype=f.dtype)
        grads = torch.autograd.grad(f, params, grad_outputs=eye, is_grads_batched=True, retain_graph=True)
        return torch.cat([g.reshape(N, -1) for g in grads], dim=1)
    except Exception:  # vmap fallback: row by row
        rows = []
        for i in range(N):
            g = torch.autograd.grad(f[i], params, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                                   for gi, p in zip(g, params)]))
        return torch.stack(rows)


def ntk_blocks(model, cfg, pts, lam_r=None, lam_0=None, n_r=400, n_0=128, n_b=50, seed=0):
    """Jacobians of the active loss-residual blocks on a random subsample.

    Returns dict name -> {"J": (n,P) tensor, "lam": (n,) weights or ones, "n_train": N_i}.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    out = {}

    idx = torch.randperm(pts.x_f.shape[0], generator=g)[:n_r]
    x = pts.x_f[idx].detach().clone().requires_grad_(True)
    t = pts.t_f[idx].detach().clone().requires_grad_(True)
    r = residual(model, x, t)
    out["r"] = {"J": _flat_jacobian(r, params),
                "lam": torch.ones(n_r, device=x.device) if lam_r is None else lam_r[idx].detach().reshape(-1),
                "n_train": pts.x_f.shape[0]}

    if cfg.use_ic:
        idx0 = torch.randperm(pts.x_0.shape[0], generator=g)[:n_0]
        e0 = pts.u_0[idx0] - model(pts.x_0[idx0], pts.t_0[idx0])
        out["ic"] = {"J": _flat_jacobian(e0, params),
                     "lam": torch.ones(len(idx0), device=x.device) if lam_0 is None else lam_0[idx0].detach().reshape(-1),
                     "n_train": pts.x_0.shape[0]}
        if lam_0 is None and cfg.ic_weight != 1.0:
            out["ic"]["lam"] = out["ic"]["lam"] * cfg.ic_weight ** 0.5

    if cfg.use_bc:
        idxb = torch.randperm(pts.t_b.shape[0], generator=g)[:n_b]
        du, dux = boundary_mismatch(model, pts.t_b[idxb].detach().clone())
        fb = torch.cat([du, dux]).reshape(-1)
        out["bc"] = {"J": _flat_jacobian(fb, params),
                     "lam": torch.ones(fb.numel(), device=x.device), "n_train": pts.t_b.shape[0]}
    return out


def kernel(blocks, weighted=True, mean_normalized=True):
    """Full effective NTK over the concatenated blocks, as a numpy array.

    weighted        : apply the self-adaptive weights, K -> Lambda K Lambda
    mean_normalized : scale block i by 1/N_i (mean-squared loss over N_i points),
                      using the subsample size as the proxy for N_i.
    """
    Js = []
    for name, b in blocks.items():
        J = b["J"]
        if weighted:
            J = b["lam"][:, None] * J
        if mean_normalized:
            J = J / np.sqrt(J.shape[0])
        Js.append(J)
    J = torch.cat(Js, dim=0).double()
    return (J @ J.T).cpu().numpy()


def block_traces(blocks, weighted=True, mean_normalized=True):
    """Trace of each diagonal block (its share of the total gradient-flow speed)."""
    out = {}
    for name, b in blocks.items():
        J = b["J"].double()
        if weighted:
            J = b["lam"][:, None].double() * J
        tr = float((J * J).sum())
        if mean_normalized:
            tr /= J.shape[0]
        out[name] = tr
    return out


def spectrum(K):
    """Eigenvalues (descending, clipped at 0) of a symmetric PSD matrix."""
    w = np.linalg.eigvalsh(K)[::-1]
    return np.clip(w, 0, None)


def spectrum_stats(w, rel_tol=1e-10):
    w = np.asarray(w)
    pos = w[w > rel_tol * w.max()]
    p = w / w.sum()
    p = p[p > 0]
    return {
        "lambda_max": float(w.max()),
        "lambda_min_pos": float(pos.min()),
        "condition": float(w.max() / pos.min()),
        "trace": float(w.sum()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "n": int(len(w)),
        "n_above_1e-3": int((w > 1e-3 * w.max()).sum()),
    }
