"""Summary table and figures from results/*.json, results/*_pred.npz, results/ntk/*.

    python scripts/plot.py            # table + all figures into figures/
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sapinn.pde import load_reference  # noqa: E402

ORDER = ["soft_uniform", "soft_sa", "hard_bc_uniform", "hard_bc_sa",
         "hard_ic_uniform", "hard_ic_sa", "hard_uniform", "hard_sa"]
LABEL = {"soft_uniform": "soft / uniform (baseline PINN)", "soft_sa": "soft / SA (paper)",
         "hard_bc_uniform": "hard BC / uniform", "hard_bc_sa": "hard BC / SA",
         "hard_ic_uniform": "hard IC / uniform", "hard_ic_sa": "hard IC / SA",
         "hard_uniform": "hard BC+IC / uniform", "hard_sa": "hard BC+IC / SA"}
COLOR = {"soft_uniform": "#7f7f7f", "soft_sa": "#1f77b4", "hard_bc_uniform": "#bcbd22", "hard_bc_sa": "#2ca02c",
         "hard_ic_uniform": "#ff9896", "hard_ic_sa": "#d62728", "hard_uniform": "#c5b0d5", "hard_sa": "#9467bd"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5})


def load_results(results):
    runs = defaultdict(list)
    for f in sorted(Path(results).glob("*_seed*.json")):
        r = json.load(open(f))
        runs[r["name"]].append(r)
    return runs


def fmt(x):
    return f"{x:.2e}"


def table(runs, path):
    rows = ["| Config | #seeds | rel. L2 (mean ± std) | best | after Adam | time/run [s] |", "|---|---|---|---|---|---|"]
    for name in [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]:
        rs = runs[name]
        e = np.array([r["rel_l2"] for r in rs])
        ea = np.array([r["rel_l2_after_adam"] for r in rs])
        rows.append(f"| {LABEL.get(name, name)} | {len(rs)} | {fmt(e.mean())} ± {fmt(e.std())} | {fmt(e.min())} "
                    f"| {fmt(ea.mean())} | {np.mean([r['time_total_s'] for r in rs]):.0f} |")
    txt = "\n".join(rows)
    Path(path).write_text(txt + "\n")
    print(txt)


def fig_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for name in [n for n in ORDER if n in runs]:
        for k, r in enumerate(runs[name]):
            h = r["history"]
            step = np.array(h["step"]); err = np.array(h["err"]); lr = np.array(h["loss_r"])
            m = ~np.isnan(err)
            kw = dict(color=COLOR.get(name, None), lw=1.1, alpha=0.9 if k == 0 else 0.35,
                      label=LABEL.get(name, name) if k == 0 else None)
            axes[0].semilogy(step[m], err[m], **kw)
            axes[1].semilogy(step, lr, **kw)
    n_adam = runs[next(iter(runs))][0]["config"]["adam_iters"]
    for ax in axes:
        ax.axvline(n_adam, color="k", lw=0.6, ls=":")
        ax.set_xlabel("iteration (Adam | L-BFGS)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative L2 error"); axes[1].set_ylabel("PDE residual loss (unweighted scale differs for SA)")
    axes[0].legend(ncol=1, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_solution(runs, results, path, names=("soft_uniform", "soft_sa", "hard_sa", "hard_uniform")):
    x, t, u = load_reference()
    names = [n for n in names if n in runs]
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(2.2 * (len(names) + 1), 4.0),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    axes[0, 0].pcolormesh(t, x, u, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    axes[0, 0].set_title("reference (Chebfun)"); axes[0, 0].set_ylabel("x"); axes[0, 0].set_xlabel("t")
    for j, tt in enumerate((0.0, 0.5, 1.0)):
        k = int(np.argmin(np.abs(t - tt)))
        axes[1, 0].plot(x, u[:, k], color=f"C{j}", lw=1, label=f"t={tt}")
    axes[1, 0].legend(); axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("u")
    for i, name in enumerate(names, start=1):
        best = min(runs[name], key=lambda r: r["rel_l2"])
        pred = np.load(Path(results) / f"{name}_seed{best['seed']}_pred.npz")["pred"]
        axes[0, i].pcolormesh(t, x, np.abs(pred - u), cmap="magma", vmin=0, vmax=0.5, shading="auto")
        axes[0, i].set_title(f"{LABEL[name]}\n|error|, L2={best['rel_l2']:.1e}", fontsize=8)
        axes[0, i].set_xlabel("t")
        for j, tt in enumerate((0.0, 0.5, 1.0)):
            k = int(np.argmin(np.abs(t - tt)))
            axes[1, i].plot(x, u[:, k], color=f"C{j}", lw=1, alpha=0.4)
            axes[1, i].plot(x, pred[:, k], color=f"C{j}", lw=1, ls="--")
        axes[1, i].set_xlabel("x")
    for ax in axes[1, 1:]:
        ax.set_ylim(-1.3, 1.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_weights(runs, results, path, name="soft_sa"):
    if name not in runs:
        return
    best = min(runs[name], key=lambda r: r["rel_l2"])
    d = np.load(Path(results) / f"{name}_seed{best['seed']}_pred.npz")
    x, t, u = load_reference()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    sc = axes[0].scatter(d["t_f"], d["x_f"], c=d["lam_r"], s=1, cmap="viridis")
    axes[0].set_title(r"self-adaptive residual weights $\lambda_r$"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
    plt.colorbar(sc, ax=axes[0])
    axes[1].plot(d["x_0"], d["lam_0"], ".", ms=2); axes[1].set_title(r"IC weights $\lambda_0(x)$"); axes[1].set_xlabel("x")
    axes[2].pcolormesh(t, x, np.abs(d["pred"] - u), cmap="magma", shading="auto")
    axes[2].set_title("|error| of the same run"); axes[2].set_xlabel("t")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_ntk(results, path, seed=0, steps=("step0", "step10000")):
    ntk_dir = Path(results) / "ntk"
    files = {n: ntk_dir / f"{n}_seed{seed}.npz" for n in ORDER if (ntk_dir / f"{n}_seed{seed}.npz").exists()}
    if not files:
        return
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(2.6 * (len(steps) + 1), 2.6))
    for ax, st in zip(axes, steps):
        for name, f in files.items():
            d = np.load(f)
            key = f"{st}_weighted_full"
            if key not in d:
                continue
            w = d[key]
            ax.semilogy(np.arange(1, len(w) + 1), w / w[0], color=COLOR[name], lw=1.1, label=LABEL[name])
        ax.set_title(f"NTK spectrum, {st.replace('step', 'iter ')}"); ax.set_xlabel("index"); ax.set_ylabel(r"$\lambda_i/\lambda_1$")
        ax.set_ylim(1e-12, 2); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    # trace share of the IC block over training (soft variants only)
    ax = axes[-1]
    for name, f in files.items():
        js = json.load(open(f.with_suffix(".json")))
        pts = []
        for st, e in js["steps"].items():
            tr = e["weighted_traces"]
            if "ic" in tr:
                pts.append((e["step"], tr["ic"] / (sum(tr.values()))))
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLOR[name], lw=1, ms=3, label=LABEL[name])
    ax.set_title("IC share of NTK trace"); ax.set_xlabel("iteration"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()
    figs = Path(a.figures); figs.mkdir(exist_ok=True)
    runs = load_results(a.results)
    if not runs:
        print("no results"); return
    table(runs, figs / "table.md")
    fig_curves(runs, figs / "curves.png")
    fig_solution(runs, a.results, figs / "solution.png")
    fig_weights(runs, a.results, figs / "weights.png")
    fig_ntk(a.results, figs / "ntk.png")
    print("figures written to", figs)


if __name__ == "__main__":
    main()
