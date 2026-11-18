"""
Run one example of the paper.

    python -m examples.run --list
    python -m examples.run disk --quick                 # smoke test (tiny settings)
    python -m examples.run disk                         # settings of the paper
    python -m examples.run ball4_windows --windows 0    # only the first window
    python -m examples.run all --quick                  # every example, smoke test

Options: --seed S (default 0), --out DIR (default examples_out/), --threads T (default 1),
--epochs/--points/--mu override the corresponding settings, --no-plots.
Output: DIR/<name>_s<seed>.json, DIR/<name>_s<seed>_loss.png, DIR/<name>_s<seed>_eigenfunctions.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pinn_eigen as pe                         # noqa: E402
from examples.configs import EXAMPLES           # noqa: E402
from examples.core import run_example           # noqa: E402


def list_examples() -> None:
    print(f"{'name':22} {'sec.':5} {'paper':42} {'nodes':>5} {'mu':>7} {'epochs':>6}  p")
    for ex in EXAMPLES.values():
        n = sum(len(w) for w in ex.windows)
        print(f"{ex.name:22} {ex.section:5} {ex.paper[:42]:42} {n:5d} {ex.mu:7g} {ex.n_epochs:6d}  {ex.p:g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Examples of 'A PINNs approach for the computation of eigenvalues'")
    ap.add_argument("name", nargs="?", help="example name, or 'all'")
    ap.add_argument("--list", action="store_true", help="list the examples")
    ap.add_argument("--quick", action="store_true", help="tiny settings (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "examples_out"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--windows", type=int, nargs="+", help="run only these windows (0-based)")
    ap.add_argument("--epochs", type=int, help="override the maximal number of epochs per node")
    ap.add_argument("--points", type=int, help="override the number of MC points per epoch")
    ap.add_argument("--mu", type=float, help="override the penalty parameter")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.name:
        list_examples()
        return 0
    names = list(EXAMPLES) if a.name == "all" else [a.name]
    for n in names:
        if n not in EXAMPLES:
            ap.error(f"unknown example {n!r}; use --list")
    pe.set_threads(a.threads)
    for n in names:
        ex = EXAMPLES[n]
        if a.windows:
            ex = replace(ex, windows=[ex.windows[i] for i in a.windows])
        if a.quick:
            ex = ex.quick()
        over = {k: v for k, v in (("n_epochs", a.epochs), ("n_samples", a.points), ("mu", a.mu)) if v is not None}
        if over:
            ex = replace(ex, **over)
        t0 = time.perf_counter()
        run_example(ex, seed=a.seed, out_dir=a.out, plots=not a.no_plots)
        print(f"[{n}] done in {time.perf_counter() - t0:.1f} s\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
