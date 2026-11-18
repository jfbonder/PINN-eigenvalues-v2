"""Per-epoch cost benchmark at paper settings (3x128, 5000 pts). python -m experiments.smoke.bench_epoch"""
import json, time, sys
from functools import partial
import torch
from experiments import common as C
import pinn_eigen as pe

CASES = {
    "2D_disk_lap": (2, pe.sample_disk, pe.boundary_disk, 2.0),
    "2D_disk_p2.2": (2, pe.sample_disk, pe.boundary_disk, 2.2),
    "3D_ball_lap": (3, pe.sample_ball_3d, pe.boundary_ball, 2.0),
    "4D_ball_lap": (4, pe.sample_ball_4d, pe.boundary_ball, 2.0),
}

def main():
    args = C.parse_args("bench_epoch")
    out = {}
    for th in (2, 4):
        pe.set_threads(th)
        for name, (d, s, b, p) in CASES.items():
            torch.manual_seed(0)
            m = pe.EigenFunctionNN(dim=d, boundary_fn=b, hidden_size=128, n_hidden=3)
            pe.train_one(m, 10.0, s, pe.V_zero, n_epochs=5, p=p, patience=10**9)  # warm-up
            _, _, info = pe.train_one(m, 10.0, s, pe.V_zero, n_epochs=60, p=p, patience=10**9, return_info=True)
            ms = 1000 * info["wall_time"] / 60
            out[f"{name}_threads{th}"] = ms
            print(f"{name:14s} threads={th}: {ms:6.1f} ms/epoch -> 3000 epochs = {3*ms:.0f}s", flush=True)
    C.save_results("bench_epoch", dict(hidden=128, n_hidden=3, n_samples=5000), timings=out)

main()
