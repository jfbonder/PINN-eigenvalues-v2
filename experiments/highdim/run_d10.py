"""High-dimension test: unit ball in R^d (default d=10), V=0, Dirichlet, warm-started scan.
Exact: E1 = j_{d/2-1,1}^2 (simple), E2 = j_{d/2,1}^2 (multiplicity d).
python -m experiments.highdim.run_d10 [--d 10] [--seed 0]   (resumable; 1 thread)"""
import argparse, json, os, time
import numpy as np, torch
from scipy.special import jn_zeros
import pinn_eigen as pe

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mu", type=float, default=200.0); ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--grid", type=str, default="51:84:3")
a = ap.parse_args()
pe.set_threads(1)
d = a.d
E1 = jn_zeros(d / 2 - 1, 1)[0] ** 2; E2 = jn_zeros(d / 2, 1)[0] ** 2
lo, hi, h = map(float, a.grid.split(":")); grid = np.arange(lo, hi + 1e-9, h)
out = f"experiments/results/highdim_d{d}_mu{int(a.mu)}_s{a.seed}.json"
def sampler(n):
    x = torch.randn(n, d); x = x / x.norm(dim=1, keepdim=True)
    return x * torch.rand(n, 1) ** (1.0 / d)
def quotients(m, n=20000):
    x = sampler(n).requires_grad_(True); u = m(x).squeeze(-1)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = sum(torch.autograd.grad(g[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(d))
    u2 = (u.detach() ** 2).mean()
    rho_var = float((g.detach() ** 2).sum(1).mean() / u2); rho_res = float(-(u.detach() * lap.detach()).mean() / u2)
    return rho_var, rho_res
rec = json.load(open(out)) if os.path.exists(out) else dict(d=d, seed=a.seed, mu=a.mu, epochs=a.epochs,
       grid=grid.tolist(), exact=dict(E1=E1, E2=E2, mult2=d), net="3x128 tanh", n_samples=5000, records=[])
torch.manual_seed(a.seed)
m = pe.EigenFunctionNN(dim=d, boundary_fn=pe.boundary_ball, hidden_size=128, n_hidden=3)
ck = out.replace(".json", ".pt")
if os.path.exists(ck): m.load_state_dict(torch.load(ck))
done = {round(r["E"], 6) for r in rec["records"]}
for E in grid:
    if round(float(E), 6) in done: continue
    _, _, info = pe.train_one(m, float(E), sampler, pe.V_zero, n_epochs=a.epochs, mu=a.mu, return_info=True)
    ev = pe.evaluate_residual(m, sampler, float(E), pe.V_zero, n_samples=20000)
    rv, rr = quotients(m)
    r = dict(E=float(E), res=float(ev["loss_res"]), mass=float(ev["mass"]), r=float(ev["r"]),
             rho_var=rv, rho_res=rr, epochs=info["epochs_run"], wall=info["wall_time"])
    rec["records"].append(r); json.dump(rec, open(out, "w"), indent=1); torch.save(m.state_dict(), ck)
    print(f"d={d} E={E:7.3f} res={r['res']:.3e} mass={r['mass']:.3f} r={r['r']:.3f} rho_var={rv:.4f} rho_res={rr:.4f} "
          f"ep={r['epochs']} t={r['wall']:.0f}s  [E1={E1:.4f} E2={E2:.4f}]", flush=True)
print("DONE", flush=True)
