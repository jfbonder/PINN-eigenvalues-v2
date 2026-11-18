"""Adaptive E-step scan (linear case): E^{i+1} = E^i + max(theta*rho_i, hmin), rho_i = sqrt((Lam_i - delta)_+),
Lam_i = normalized loss at E^i (fresh sample). Since Lam(E) = dist(E,sigma)^2 up to the training error delta,
no eigenvalue lies in (E^i - rho_i, E^i + rho_i). Unit square, V=0, same settings as the uniform offset scans.
python -m experiments.mu_cost.adaptive --tag T --theta 1 --hmin 2.5 --delta 1 [--seed 0]   (resumable)"""
import argparse, json, os, time, numpy as np, torch
import pinn_eigen as pe
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--theta", type=float, default=1.0)
ap.add_argument("--hmin", type=float, default=2.5); ap.add_argument("--delta", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mu", type=float, default=450.0)
ap.add_argument("--E0", type=float, default=15.0); ap.add_argument("--E1", type=float, default=105.0)
a = ap.parse_args(); pe.set_threads(1)
out = f"experiments/results/mucost_adapt_{a.tag}.json"; ck = out.replace(".json", ".pt")
sampler = pe.sample_square; bfun = pe.boundary_square
def quot(m, n=40000):
    x = sampler(n).requires_grad_(True); u = m(x).squeeze(-1)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = sum(torch.autograd.grad(g[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(2))
    u = u.detach(); return float(-(u * lap.detach()).mean() / (u ** 2).mean())
rec = json.load(open(out)) if os.path.exists(out) else dict(config=vars(a), records=[])
torch.manual_seed(a.seed)
m = pe.EigenFunctionNN(dim=2, boundary_fn=bfun, hidden_size=128, n_hidden=3)
if os.path.exists(ck): m.load_state_dict(torch.load(ck))
E = rec["records"][-1]["E_next"] if rec["records"] else a.E0
while E <= a.E1 + 1e-9:
    t0 = time.time()
    _, _, info = pe.train_one(m, E, sampler, pe.V_zero, n_epochs=3000, mu=a.mu, return_info=True)
    ev = pe.evaluate_residual(m, sampler, E, pe.V_zero, n_samples=20000)
    Lam = ev["r"] ** 2; rho = np.sqrt(max(Lam - a.delta, 0.0)); step = max(a.theta * rho, a.hmin)
    r = dict(E=E, r=ev["r"], mass=ev["mass"], res=ev["loss_res"], rho=rho, step=step, E_next=E + step,
             rq=quot(m), epochs=info["epochs_run"], wall=time.time() - t0)
    rec["records"].append(r); json.dump(rec, open(out, "w"), indent=1); torch.save(m.state_dict(), ck)
    print(f"[{a.tag}] E={E:7.3f} r={r['r']:.3f} mass={r['mass']:.3f} rq={r['rq']:.4f} step={step:.2f} ep={r['epochs']} t={r['wall']:.0f}s", flush=True)
    E += step
print("DONE", sum(x["wall"] for x in rec["records"]), len(rec["records"]), flush=True)
