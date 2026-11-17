"""PINN loss-curve windows (local E-sweeps) for the FEM comparison. Resumable.

usage: python -m experiments.fem.pinn_windows CASE --centers 9.58 32.8 [--half 1.5 --step 0.5] [--seed 0]
Each window = chain of warm-started trainings on E = c-half, ..., c+half (first node: fresh network).
Settings follow the paper/notebook: 3x128 tanh, Adam lr 1e-3, early stop patience 300 tol 1e-5,
fresh MC batch each epoch; per-case mu, n_epochs, n_samples as in FBS_PINN_notebook_v2.ipynb.
Records per E (saved incrementally in experiments/results/fem_pinn_<case>_s<seed>.json):
  loss_res (last training batch), eval_* on fresh 20000 pts, r2 = mean(res^2)/mean(u^2) (normalised Lambda),
  rho_var = (mean|grad u|^2 + V u^2)/mean u^2 and rho_res = -mean(u (Lap u - V u))/mean(u^2) on a fixed
  100000-pt sample (Rayleigh quotients of the trained net).
"""
import argparse, json, time, os, sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
import pinn_eigen as pe
from experiments import common as C

CASES = {
    "double_well": dict(dim=3, sampler=pe.sample_ball_3d, V=partial(pe.V_double_well_gaussian, a=1.0, sigma=0.5, V0=1.0),
                        B=pe.boundary_ball, mu=400.0, n_epochs=2000, n_samples=5000),
    "four_well": dict(dim=3, sampler=pe.sample_ball_3d, V=pe.V_four_well_bump, B=pe.boundary_ball,
                      mu=1000.0, n_epochs=2000, n_samples=2000),
    "harm_w10": dict(dim=2, sampler=partial(pe.sample_disk, radius=1.0), V=partial(pe.V_harmonic, omega=10.0),
                     B=pe.boundary_disk, mu=100.0, n_epochs=3000, n_samples=5000),
    "harm_w1": dict(dim=2, sampler=partial(pe.sample_disk, radius=1.0), V=partial(pe.V_harmonic, omega=1.0),
                    B=pe.boundary_disk, mu=100.0, n_epochs=3000, n_samples=5000),
    "ball_free": dict(dim=3, sampler=pe.sample_ball_3d, V=pe.V_zero, B=pe.boundary_ball,
                      mu=200.0, n_epochs=2000, n_samples=5000),
}

def rayleigh(model, sampler, V_fn, n=100000, chunk=10000, seed=12345):
    g = torch.Generator().manual_seed(seed)
    st = torch.get_rng_state(); torch.manual_seed(seed)
    s_u2 = s_g = s_vu = s_ulu = 0.0; tot = 0
    for _ in range(n // chunk):
        x = sampler(chunk).requires_grad_(True)
        u = model(x)
        (gu,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        lap = pe._laplacian(u, x)
        V = V_fn(x).detach()
        s_u2 += (u ** 2).sum().item(); s_g += (gu ** 2).sum().item()
        s_vu += (V * u ** 2).sum().item(); s_ulu += (u * (lap - V * u)).sum().item()
        tot += chunk
    torch.set_rng_state(st)
    return (s_g + s_vu) / s_u2, -s_ulu / s_u2, s_u2 / tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("--centers", type=float, nargs="+", required=True)
    ap.add_argument("--half", type=float, default=1.5); ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = CASES[a.case]
    ne = a.epochs or cfg["n_epochs"]
    path = C.RESULTS_DIR / f"fem_pinn_{a.case}{a.tag}_s{a.seed}.json"
    data = json.load(open(path)) if path.exists() else dict(
        case=a.case, seed=a.seed, settings=dict(mu=cfg["mu"], n_epochs=ne, n_samples=cfg["n_samples"], lr=1e-3,
        patience=300, tol=1e-5, hidden=128, n_hidden=3, vol=1.0, step=a.step, half=a.half), windows={})
    mf = lambda: pe.EigenFunctionNN(dim=cfg["dim"], boundary_fn=cfg["B"])
    for c in a.centers:
        key = f"{c:.4f}"
        w = data["windows"].setdefault(key, dict(center=c, records=[]))
        Es = [round(c + (i - round(a.half / a.step)) * a.step, 6) for i in range(2 * round(a.half / a.step) + 1)]
        done = {round(r["E"], 6) for r in w["records"]}
        torch.manual_seed(a.seed * 1000 + int(round(c * 100)))
        model = None
        for i, E in enumerate(Es):
            if round(E, 6) in done:
                continue
            # (resume simplification: a resumed window restarts with a fresh network)
            if model is None or i == 0 or not w["records"]:
                model = mf()
            t0 = time.perf_counter()
            res, norm, info = pe.train_one(model, E, cfg["sampler"], cfg["V"], n_samples=cfg["n_samples"],
                                           n_epochs=ne, mu=cfg["mu"], return_info=True)
            ev = pe.evaluate_residual(model, cfg["sampler"], E, cfg["V"], n_samples=20000)
            rv, rr, m2 = rayleigh(model, cfg["sampler"], cfg["V"])
            rec = dict(E=E, loss_res=res, norm_err=norm, epochs_run=info["epochs_run"], train_time=info["wall_time"],
                       eval_loss_res=ev["loss_res"], eval_mass=ev["mass"], r=ev["r"], r2=ev["r"] ** 2,
                       rho_var=rv, rho_res=rr, mean_u2=m2, wall=time.perf_counter() - t0)
            w["records"].append(rec)
            json.dump(data, open(path, "w"), indent=1)
            print(f"{a.case} s{a.seed} c={c} E={E:7.3f} res={res:.3e} r2={rec['r2']:.3e} rho_var={rv:.4f} rho_res={rr:.4f} "
                  f"ep={info['epochs_run']} t={rec['wall']:.0f}s", flush=True)
            # warm start next
            nm = mf(); nm.load_state_dict(model.state_dict()); model = nm

if __name__ == "__main__":
    main()
