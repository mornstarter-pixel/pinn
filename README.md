# Self-Adaptive PINNs with hard constraints (PyTorch)

PyTorch reimplementation of **Self-Adaptive Physics-Informed Neural Networks**
(L. McClenny, U. Braga-Neto, *J. Comput. Phys.* 474, 2023,
[arXiv:2009.04544](https://arxiv.org/abs/2009.04544); official TensorFlow code
[levimcclenny/SA-PINNs](https://github.com/levimcclenny/SA-PINNs)) on the stiff
1D Allen-Cahn benchmark, extended with **exact (hard) enforcement of the periodic
boundary and initial conditions** and an **empirical NTK analysis**.

```
u_t - 1e-4 u_xx + 5u^3 - 5u = 0,   x in [-1,1], t in [0,1]
u(x,0) = x^2 cos(pi x),   u(-1,t) = u(1,t),   u_x(-1,t) = u_x(1,t)
```

## What is implemented

| component | paper / official code | this repo |
|---|---|---|
| framework | TensorFlow 2.3 + custom eager L-BFGS | PyTorch (`torch.optim.LBFGS`, strong-Wolfe) |
| network | [2,128,128,128,128,1], tanh, Glorot normal | same |
| points | 20 000 LHS collocation, 512 IC, 100 BC times | same |
| self-adaptive weights | per-point `lambda`, loss `mean((lambda*r)^2)`, Adam (lr 5e-3, beta1 .99) with gradient ascent, frozen during L-BFGS | same (`maximize=True`) |
| schedule | 10k Adam + 10k L-BFGS | same |
| constraints | soft IC + soft periodic BC | `soft`, `hard_bc`, `hard_ic`, `hard` |
| analysis | - | empirical NTK spectra / block traces at checkpoints |

Hard constraints (`sapinn/model.py`):

* **periodic BC**: the input `x` is replaced by `(cos k pi x, sin k pi x)`, so every
  network output is C-infinity periodic on [-1,1] and the BC loss is removed.
* **IC**: `u(x,t) = u0(x) + t * N(x,t)` so that `u(x,0) = u0(x)` exactly and the
  IC loss is removed.  Note that `u0` is only C0-periodic (`u0'(-1) = 2`,
  `u0'(1) = -2`), so with both constraints the ansatz carries a fixed jump of
  `u_x` across the periodic boundary that the network cannot remove; see report.

## Run

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt                       # CUDA build of torch recommended
python scripts/run_ablation.py --constraints soft hard_bc hard_ic hard --weightings uniform sa --seeds 0 1 2
python scripts/run_ntk.py          # NTK of the saved checkpoints
python scripts/plot.py             # figures/table.md, curves.png, solution.png, weights.png, ntk.png
```

`data/AC.mat` is the Chebfun reference solution from
[maziarraissi/PINNs](https://github.com/maziarraissi/PINNs), as used by the paper.

## Layout

```
sapinn/pde.py     problem, reference data, sampling, relative L2 error
sapinn/model.py   MLP with optional Fourier embedding (hard BC) and IC ansatz (hard IC)
sapinn/train.py   Config + training loop (Adam min-max phase, L-BFGS phase, logging, checkpoints)
sapinn/ntk.py     empirical NTK of the loss blocks, weighted by lambda
scripts/          run_ablation.py, run_ntk.py, plot.py
results/          one JSON + NPZ per run, checkpoints under results/ckpt, NTK under results/ntk
```
