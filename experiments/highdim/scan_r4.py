"""Full warm-started scan in the unit ball of R^4 (paper Fig. loss3D right) with the corrected uniform sampler.
usage: python -m experiments.highdim.scan_r4 [--seed 0]  -> experiments/results/scan_r4_s<seed>.json (resumable)"""
import argparse, json, time
import numpy as np, torch
import pinn_eigen as pe
from experiments import common as C
from experiments.fem.mc_eval import mc_eval
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--step", type=float, default=1.5)
ap.add_argument("--E0", type=float, default=0.7); ap.add_argument("--E1", type=float, default=55.0); a = ap.parse_args()
torch.set_num_threads(1)
Es = [round(a.E0 + i * a.step, 4) for i in range(int((a.E1 - a.E0) / a.step) + 1)]
path = C.RESULTS_DIR / f"scan_r4_s{a.seed}.json"
data = json.load(open(path)) if path.exists() else dict(settings=dict(mu=200.0, n_epochs=2000, n_samples=5000, step=a.step, seed=a.seed, warm=True), records=[])
torch.manual_seed(a.seed)
model = pe.EigenFunctionNN(dim=4, boundary_fn=pe.boundary_ball)
ck = C.RESULTS_DIR / f"scan_r4_s{a.seed}_last.pt"
if data["records"]: model.load_state_dict(torch.load(ck))
done = {r["E"] for r in data["records"]}
for E in Es:
    if E in done: continue
    t0 = time.perf_counter()
    res, norm, info = pe.train_one(model, E, pe.sample_ball_4d, pe.V_zero, n_samples=5000, n_epochs=2000, mu=200.0, return_info=True)
    ev = mc_eval(model, pe.sample_ball_4d, pe.V_zero, E)
    data["records"].append(dict(E=E, loss_res=res, epochs_run=info["epochs_run"], wall=time.perf_counter() - t0, **ev))
    json.dump(data, open(path, "w"), indent=1); torch.save(model.state_dict(), ck)
    print(f"E={E:7.3f} loss_res={res:.4f} rho={ev['rho_res']:.4f} r={ev['r']:.3f} mass={ev['mass']:.3f} ep={info['epochs_run']} t={time.perf_counter()-t0:.0f}s", flush=True)
