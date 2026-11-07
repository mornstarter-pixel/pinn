"""NTK analysis of saved checkpoints.

For every (config, seed) under results/ckpt and every checkpoint step, compute
the empirical NTK of the active loss blocks and store eigenvalue spectra,
block traces and summary statistics in results/ntk/<name>_seed<s>.npz / .json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sapinn.ntk import block_traces, kernel, ntk_blocks, spectrum, spectrum_stats  # noqa: E402
from sapinn.pde import Points  # noqa: E402
from sapinn.train import Config, build_model  # noqa: E402


def analyse(result_json, ckpt_dir, out_dir, n_r, n_0, n_b, device):
    res = json.load(open(result_json))
    cfg_d = dict(res["config"])
    cfg_d["hidden"] = tuple(cfg_d["hidden"])
    cfg_d["ckpt_steps"] = tuple(cfg_d["ckpt_steps"])
    cfg_d["device"] = device
    cfg = Config(**cfg_d)
    dtype = torch.float64  # analyse in double precision regardless of training dtype
    pts = Points(cfg.n_f, cfg.n_b, cfg.seed, device, dtype)

    summary = {"name": cfg.name, "seed": cfg.seed, "steps": {}}
    spectra = {}
    for ck in sorted(ckpt_dir.glob("*.pt"), key=lambda p: (p.stem != "step0", p.stem)):
        state = torch.load(ck, map_location=device)
        model = build_model(cfg)
        model.load_state_dict(state["model"])
        model.to(dtype)
        lam_r = None if state["lam_r"] is None else state["lam_r"].to(device=device, dtype=dtype)
        lam_0 = None if state["lam_0"] is None else state["lam_0"].to(device=device, dtype=dtype)
        blocks = ntk_blocks(model, cfg, pts, lam_r, lam_0, n_r=n_r, n_0=n_0, n_b=n_b, seed=cfg.seed)

        entry = {"step": int(state["step"])}
        for weighted in (False, True):
            tag = "weighted" if weighted else "unweighted"
            K = kernel(blocks, weighted=weighted)
            w = spectrum(K)
            spectra[f"{ck.stem}_{tag}_full"] = w
            entry[f"{tag}_full"] = spectrum_stats(w)
            entry[f"{tag}_traces"] = block_traces(blocks, weighted=weighted)
            for name in blocks:
                wb = spectrum(kernel({name: blocks[name]}, weighted=weighted))
                spectra[f"{ck.stem}_{tag}_{name}"] = wb
                entry[f"{tag}_{name}"] = spectrum_stats(wb)
        summary["steps"][ck.stem] = entry
        print(f"  {cfg.name} s{cfg.seed} {ck.stem:9s} "
              f"cond(unw)={entry['unweighted_full']['condition']:.2e} "
              f"cond(w)={entry['weighted_full']['condition']:.2e} "
              f"traces(w)={ {k: f'{v:.2e}' for k, v in entry['weighted_traces'].items()} }", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{cfg.name}_seed{cfg.seed}.npz", **spectra)
    json.dump(summary, open(out_dir / f"{cfg.name}_seed{cfg.seed}.json", "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--names", nargs="*", default=None, help="e.g. soft_sa_seed0; default: all")
    p.add_argument("--n_r", type=int, default=400)
    p.add_argument("--n_0", type=int, default=128)
    p.add_argument("--n_b", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    results = Path(a.results)
    out = results / "ntk"
    for js in sorted(results.glob("*_seed*.json")):
        name = js.stem
        if a.names and name not in a.names:
            continue
        ckpt_dir = results / "ckpt" / name
        if not ckpt_dir.exists():
            continue
        if (out / f"{name}.json").exists() and not a.force:
            print(f"skip {name}")
            continue
        print(f"NTK: {name}")
        analyse(js, ckpt_dir, out, a.n_r, a.n_0, a.n_b, a.device)


if __name__ == "__main__":
    main()
