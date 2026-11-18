"""Round-2 PINN runs (1 core, resumable). usage: python -m experiments.fem.pinn_round2 TASK [--seed 0]
TASK: ext (harmonic w=10 window extended to 49.2), tri / ball3 / ball4 (paper Tables 4 / 2 / 3: 3 offset nodes per listed eigenvalue,
paper settings), scan (blind double-well scan [4.7,49.7] step 1.5 -> offset from the FEM spectrum).
Results: experiments/results/fem_r2_<task>_s<seed>.json with per-node MC-evaluated quantities (+ standard errors)."""
import argparse, json, time, os
from functools import partial
from pathlib import Path
import numpy as np, torch
import pinn_eigen as pe
from experiments import common as C
from experiments.fem.mc_eval import mc_eval

A = lambda E, offs=(-0.9, 0.1, 1.1): [round(E + o, 4) for o in offs]
dw = partial(pe.V_double_well_gaussian, a=1.0, sigma=0.5, V0=1.0)
TASKS = {
  "ext": dict(dim=2, B=pe.boundary_disk, sampler=partial(pe.sample_disk, radius=1.0), V=partial(pe.V_harmonic, omega=10.0), mu=100.0, n_epochs=3000, n_samples=5000,
              windows={"ext_45.2-49.2": [round(45.2 + 0.5 * i, 3) for i in range(9)]}),
  "tri": dict(dim=2, B=pe.boundary_triangle, sampler=pe.sample_triangle, V=pe.V_zero, mu=15000.0, n_epochs=2000, n_samples=4000,
              windows={f"{k:.4f}": A(k) for k in [5 * np.pi**2, 10 * np.pi**2, 13 * np.pi**2, 17 * np.pi**2]}),
  "ball3": dict(dim=3, B=pe.boundary_ball, sampler=pe.sample_ball_3d, V=pe.V_zero, mu=100.0, n_epochs=2000, n_samples=5000,
              windows={f"{k:.4f}": A(k) for k in [9.8696, 20.1907, 33.2175, 39.4784]}),
  "ball4": dict(dim=4, B=pe.boundary_ball, sampler=pe.sample_ball_4d, V=pe.V_zero, mu=200.0, n_epochs=2000, n_samples=5000,
              windows={f"{k:.4f}": A(k) for k in [14.6820, 26.3746]}),
  "scan": dict(dim=3, B=pe.boundary_ball, sampler=pe.sample_ball_3d, V=dw, mu=400.0, n_epochs=2000, n_samples=5000,
              windows={"scan_4.7-49.7": [round(4.7 + 1.5 * i, 3) for i in range(31)]}),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1)
    T = TASKS[a.task]
    path = C.RESULTS_DIR / f"fem_r2_{a.task}_s{a.seed}.json"
    data = json.load(open(path)) if path.exists() else dict(task=a.task, seed=a.seed, settings={k: T[k] for k in ("mu", "n_epochs", "n_samples")}, windows={})
    mf = lambda: pe.EigenFunctionNN(dim=T["dim"], boundary_fn=T["B"])
    for key, Es in T["windows"].items():
        recs = data["windows"].setdefault(key, [])
        done = {round(r["E"], 4) for r in recs}
        torch.manual_seed(a.seed * 7919 + abs(hash(key)) % 1000 if False else a.seed * 7919 + int(float(key.split("_")[-1].split("-")[0]) * 10))
        model = None
        for i, E in enumerate(Es):
            if E in done: continue
            if model is None or i == 0: model = mf()
            t0 = time.perf_counter()
            res, norm, info = pe.train_one(model, E, T["sampler"], T["V"], n_samples=T["n_samples"], n_epochs=T["n_epochs"], mu=T["mu"], return_info=True)
            ev = mc_eval(model, T["sampler"], T["V"], E)
            rec = dict(E=E, loss_res=res, epochs_run=info["epochs_run"], train_time=info["wall_time"], r2=ev["r"] ** 2, wall=time.perf_counter() - t0, **ev)
            recs.append(rec); json.dump(data, open(path, "w"), indent=1)
            print(f"{a.task} s{a.seed} {key} E={E:8.3f} rho={ev['rho_res']:.4f}+-{ev['rho_res_se']:.4f} r={ev['r']:.3f}+-{ev['r_se']:.3f} mass={ev['mass']:.3f} ep={info['epochs_run']} t={rec['wall']:.0f}s", flush=True)
            nm = mf(); nm.load_state_dict(model.state_dict()); model = nm

if __name__ == "__main__":
    main()
