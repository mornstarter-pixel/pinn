"""Training loop: {soft, hard_bc, hard_ic, hard} x {uniform, sa} on Allen-Cahn.

Follows the official AC.py: full-batch Adam (lr 5e-3, beta1 0.99) for the
network and a second Adam doing gradient *ascent* on the per-point weights,
then L-BFGS on the network only with the weights frozen.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import PINN
from .pde import Points, Reference, boundary_mismatch, residual


@dataclass
class Config:
    constraint: str = "soft"      # soft | hard_bc | hard_ic | hard
    weighting: str = "uniform"    # uniform | sa
    seed: int = 0
    n_f: int = 20000
    n_b: int = 100
    hidden: tuple = (128, 128, 128, 128)
    n_fourier: int = 1
    ic_scale: str = "linear"
    adam_iters: int = 10000
    lbfgs_iters: int = 10000
    lr: float = 5e-3
    lr_lambda: float = 5e-3
    beta1: float = 0.99
    ic_weight: float = 1.0        # fixed IC weight when weighting == uniform
    lambda_r_init: float = 1.0    # lambda_r ~ lambda_r_init * U[0,1)
    lambda_0_init: float = 100.0  # lambda_0 ~ lambda_0_init * U[0,1)
    dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    eval_every: int = 500
    ckpt_steps: tuple = (0, 1000, 5000, 10000)
    tag: str = ""

    @property
    def use_ic(self):
        return self.constraint in ("soft", "hard_bc")

    @property
    def use_bc(self):
        return self.constraint in ("soft", "hard_ic")

    @property
    def name(self):
        s = f"{self.constraint}_{self.weighting}"
        if self.weighting == "uniform" and self.ic_weight != 1.0:
            s += f"_w{self.ic_weight:g}"
        if self.constraint in ("hard_bc", "hard") and self.n_fourier != 1:
            s += f"_K{self.n_fourier}"
        if self.tag:
            s += f"_{self.tag}"
        return s


def build_model(cfg):
    torch.manual_seed(cfg.seed)
    model = PINN(hidden=cfg.hidden, periodic=cfg.constraint in ("hard_bc", "hard"),
                 n_fourier=cfg.n_fourier, hard_ic=cfg.constraint in ("hard_ic", "hard"),
                 ic_scale=cfg.ic_scale)
    return model.to(device=cfg.device, dtype=getattr(torch, cfg.dtype))


def build_lambdas(cfg, pts):
    """Self-adaptive weights (None when not used)."""
    lam_r = lam_0 = None
    if cfg.weighting == "sa":
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 12345)
        kw = dict(device=pts.x_f.device, dtype=pts.x_f.dtype)
        lam_r = (cfg.lambda_r_init * torch.rand(pts.x_f.shape, generator=g)).to(**kw).requires_grad_(True)
        if cfg.use_ic:
            lam_0 = (cfg.lambda_0_init * torch.rand(pts.x_0.shape, generator=g)).to(**kw).requires_grad_(True)
    return lam_r, lam_0


def loss_terms(model, pts, cfg, lam_r=None, lam_0=None):
    """Returns (loss_r, loss_0, loss_b); unused terms are zero tensors."""
    zero = torch.zeros((), device=pts.x_f.device, dtype=pts.x_f.dtype)
    r = residual(model, pts.x_f, pts.t_f)
    loss_r = ((lam_r * r) ** 2).mean() if lam_r is not None else (r ** 2).mean()
    loss_0 = zero
    if cfg.use_ic:
        e0 = pts.u_0 - model(pts.x_0, pts.t_0)
        loss_0 = ((lam_0 * e0) ** 2).mean() if lam_0 is not None else cfg.ic_weight * (e0 ** 2).mean()
    loss_b = zero
    if cfg.use_bc:
        du, dux = boundary_mismatch(model, pts.t_b)
        loss_b = (du ** 2).mean() + (dux ** 2).mean()
    return loss_r, loss_0, loss_b


def save_ckpt(path, model, lam_r, lam_0, step):
    torch.save({"model": model.state_dict(), "step": step,
                "lam_r": None if lam_r is None else lam_r.detach().cpu(),
                "lam_0": None if lam_0 is None else lam_0.detach().cpu()}, path)


def run(cfg: Config, out_dir="results", verbose=True):
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "ckpt" / f"{cfg.name}_seed{cfg.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, cfg.dtype)

    pts = Points(cfg.n_f, cfg.n_b, cfg.seed, cfg.device, dtype)
    pts.x_f.requires_grad_(True)
    pts.t_f.requires_grad_(True)
    ref = Reference(cfg.device, dtype)
    model = build_model(cfg)
    lam_r, lam_0 = build_lambdas(cfg, pts)
    lams = [l for l in (lam_r, lam_0) if l is not None]

    hist = {"step": [], "loss_r": [], "loss_0": [], "loss_b": [], "err": [], "phase": []}
    state = {"step": 0, "t0": time.time()}

    def log(lr_, l0_, lb_, phase, force=False):
        s = state["step"]
        if s % cfg.log_every == 0 or force:
            err = ref.rel_l2(model) if (s % cfg.eval_every == 0 or force) else float("nan")
            hist["step"].append(s)
            hist["phase"].append(phase)
            hist["loss_r"].append(float(lr_))
            hist["loss_0"].append(float(l0_))
            hist["loss_b"].append(float(lb_))
            hist["err"].append(err)
            if verbose and (s % (cfg.eval_every * 2) == 0 or force):
                print(f"[{cfg.name} s{cfg.seed}] {phase:5s} {s:6d}  Lr={lr_:.2e} L0={l0_:.2e} Lb={lb_:.2e} "
                      f"err={err:.3e}  {time.time() - state['t0']:.0f}s", flush=True)

    # ---------------- Adam phase (min over theta, max over lambda) -------------
    opt_net = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_lam = torch.optim.Adam(lams, lr=cfg.lr_lambda, betas=(cfg.beta1, 0.999), maximize=True) if lams else None
    if 0 in cfg.ckpt_steps:
        save_ckpt(ckpt_dir / "step0.pt", model, lam_r, lam_0, 0)
    for it in range(cfg.adam_iters):
        opt_net.zero_grad(set_to_none=True)
        if opt_lam:
            opt_lam.zero_grad(set_to_none=True)
        lr_, l0_, lb_ = loss_terms(model, pts, cfg, lam_r, lam_0)
        (lr_ + l0_ + lb_).backward()
        opt_net.step()
        if opt_lam:
            opt_lam.step()
        state["step"] = it + 1
        log(lr_.item(), l0_.item(), lb_.item(), "adam")
        if state["step"] in cfg.ckpt_steps:
            save_ckpt(ckpt_dir / f"step{state['step']}.pt", model, lam_r, lam_0, state["step"])
    t_adam = time.time() - state["t0"]
    err_adam = ref.rel_l2(model)

    # ---------------- L-BFGS phase (lambda frozen) -----------------------------
    for l in lams:
        l.requires_grad_(False)
    if cfg.lbfgs_iters > 0:
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                                max_eval=int(1.25 * cfg.lbfgs_iters), history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
        base = state["step"]

        def closure():
            opt.zero_grad(set_to_none=True)
            lr_, l0_, lb_ = loss_terms(model, pts, cfg, lam_r, lam_0)
            loss = lr_ + l0_ + lb_
            loss.backward()
            state["step"] += 1
            log(lr_.item(), l0_.item(), lb_.item(), "lbfgs")
            return loss

        opt.step(closure)
        state["n_lbfgs_evals"] = state["step"] - base
    t_total = time.time() - state["t0"]

    lr_, l0_, lb_ = loss_terms(model, pts, cfg, lam_r, lam_0)
    log(lr_.item(), l0_.item(), lb_.item(), "final", force=True)
    save_ckpt(ckpt_dir / "final.pt", model, lam_r, lam_0, state["step"])

    pred = ref.predict(model)
    result = {
        "config": asdict(cfg), "name": cfg.name, "seed": cfg.seed,
        "rel_l2": ref.rel_l2(model), "rel_l2_after_adam": err_adam,
        "slice_errors": ref.slice_errors(model),
        "final_loss": {"r": lr_.item(), "ic": l0_.item(), "bc": lb_.item()},
        "time_adam_s": t_adam, "time_total_s": t_total,
        "n_lbfgs_evals": state.get("n_lbfgs_evals", 0), "n_params": model.n_params(),
        "history": hist,
    }
    with open(out_dir / f"{cfg.name}_seed{cfg.seed}.json", "w") as f:
        json.dump(result, f)
    np.savez_compressed(out_dir / f"{cfg.name}_seed{cfg.seed}_pred.npz", pred=pred,
                        x_f=pts.x_f.detach().cpu().numpy(), t_f=pts.t_f.detach().cpu().numpy(),
                        lam_r=np.array([]) if lam_r is None else lam_r.detach().cpu().numpy(),
                        x_0=pts.x_0.cpu().numpy(),
                        lam_0=np.array([]) if lam_0 is None else lam_0.detach().cpu().numpy())
    if verbose:
        print(f"==> {cfg.name} seed {cfg.seed}: rel L2 = {result['rel_l2']:.3e} "
              f"(after Adam {err_adam:.3e}), {t_total:.0f}s", flush=True)
    return result
