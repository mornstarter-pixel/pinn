"""Run the {constraint} x {weighting} x {seed} ablation on Allen-Cahn.

Example:
    python scripts/run_ablation.py --constraints soft hard --weightings uniform sa --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sapinn.train import Config, run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--constraints", nargs="+", default=["soft", "hard_bc", "hard_ic", "hard"])
    p.add_argument("--weightings", nargs="+", default=["uniform", "sa"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--adam_iters", type=int, default=10000)
    p.add_argument("--lbfgs_iters", type=int, default=10000)
    p.add_argument("--n_fourier", type=int, default=1)
    p.add_argument("--ic_weight", type=float, default=1.0)
    p.add_argument("--ic_scale", default="linear")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="results")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in a.constraints:
        for w in a.weightings:
            for s in a.seeds:
                cfg = Config(constraint=c, weighting=w, seed=s, adam_iters=a.adam_iters,
                             lbfgs_iters=a.lbfgs_iters, n_fourier=a.n_fourier, ic_weight=a.ic_weight,
                             ic_scale=a.ic_scale, dtype=a.dtype, tag=a.tag)
                f = out / f"{cfg.name}_seed{s}.json"
                if f.exists() and not a.force:
                    r = json.load(open(f))
                    print(f"skip {cfg.name} seed {s}: rel L2 = {r['rel_l2']:.3e}")
                    continue
                run(cfg, out)


if __name__ == "__main__":
    main()
