"""
experiments/common.py
=====================
Shared helpers for the revision experiments.

Usage pattern (run from the repo root):

    python -m experiments.<pkg>.<script> [--threads 2] [--seed 0] [--name NAME]

A script typically does

    from experiments import common as C
    args = C.parse_args(default_name="my_exp")          # --threads --seed --name --out
    C.setup(args)                                        # seeds + torch threads
    with C.Timer() as t:
        residuals, models, records = pe.sweep(..., seed=args.seed, return_records=True)
    C.save_results(args.name, config=cfg, records=records,
                   eigenvalues=eigs, timings={"total_s": t.elapsed})
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

if str(REPO_ROOT) not in sys.path:          # so `import pinn_eigen` works from anywhere
    sys.path.insert(0, str(REPO_ROOT))


def seed_everything(seed: int) -> None:
    """Seed python `random`, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_threads(n: int) -> int:
    torch.set_num_threads(int(n))
    return torch.get_num_threads()


class Timer:
    """Context manager / stopwatch:  with Timer() as t: ...;  t.elapsed (seconds)."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.t1: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        self.t1 = None
        return self

    def __exit__(self, *exc) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return (self.t1 if self.t1 is not None else time.perf_counter()) - self.t0


def parse_args(default_name: str, argv=None, extra=None) -> argparse.Namespace:
    """Standard CLI: --threads, --seed, --name, --quick. `extra(parser)` may add arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=int(os.environ.get("PINN_THREADS", 2)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default=default_name)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    if extra is not None:
        extra(ap)
    return ap.parse_args(argv)


def setup(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    set_threads(args.threads)


def _jsonable(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    if callable(o):
        return getattr(o, "__name__", repr(o))
    return repr(o)


def save_results(name: str, config: dict, records: Optional[list] = None,
                 eigenvalues: Optional[list] = None, timings: Optional[dict] = None,
                 extra: Optional[dict] = None, out_dir: Optional[Path] = None) -> Path:
    """Write experiments/results/<name>.json and return its path."""
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        name=name,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        host=dict(hostname=socket.gethostname(), threads=torch.get_num_threads(),
                  torch=torch.__version__, python=platform.python_version(),
                  cpu_count=os.cpu_count()),
        config=config,
        records=records or [],
        eigenvalues=eigenvalues or [],
        timings=timings or {},
    )
    if extra:
        payload.update(extra)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=_jsonable)
    print(f"saved {path}", flush=True)
    return path


def load_results(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)
