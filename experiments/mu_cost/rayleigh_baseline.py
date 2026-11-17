"""Item 5: Rayleigh-quotient PINN with sequential (penalty) deflation on the unit square.
min_u  int|grad u|^2/int u^2 + beta * sum_{j<k} (int u u_j)^2 / (int u^2 int u_j^2)
Same MLP (3x128 tanh), B(x), Adam lr 1e-3, 5000 fresh points/epoch, same epochs and early stopping.
Eigenvalue reported = closed-form quotient (grad form) on a fixed 200k sample and on a 300^2 midpoint grid.
Usage: python -m experiments.mu_cost.rayleigh_baseline --seed 0 --K 6 --epochs 3000 [--tag name]"""
import argparse, time
import numpy as np, torch
from experiments.mu_cost import lib as L
import pinn_eigen as pe

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, default=6)
ap.add_argument("--epochs", type=int, default=3000)
ap.add_argument("--beta", type=float, default=200.0)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--tag", type=str, default=None)
a = ap.parse_args()
torch.set_num_threads(1)
tag = a.tag or f"mucost_rq_s{a.seed}_e{a.epochs}"
path = L.RES / f"{tag}.json"
data = L.load_json(path, dict(tag=tag, config=dict(seed=a.seed, K=a.K, epochs=a.epochs, beta=a.beta, hidden=a.hidden,
                    n_samples=5000, lr=1e-3, patience=300, tol=1e-5, threads=1), records=[]))
exact = L.exact_eigs(a.K)
models = []
for rec in data["records"]:
    m = L.make_model(a.hidden); m.load_state_dict(torch.load(L.CK / f"{tag}_{rec['k']}.pt")); m.eval(); models.append(m)

def grad_sq(model, x):
    x = x.clone().requires_grad_(True)
    u = model(x)
    (g,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
    return u, (g ** 2).sum(1, keepdim=True)

for k in range(len(models), a.K):
    torch.manual_seed(a.seed * 1000 + 100 + k)
    model = L.make_model(a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, noimp, t0 = float("inf"), 0, time.perf_counter()
    ep = 0
    for ep in range(1, a.epochs + 1):
        x = L.sampler(5000)
        opt.zero_grad()
        u, g2 = grad_sq(model, x)
        den = (u ** 2).mean()
        q = g2.mean() / den
        pen = 0.0
        for mj in models:
            with torch.no_grad():
                uj = mj(x)
            pen = pen + (u * uj).mean() ** 2 / (den * (uj ** 2).mean())
        loss = q + a.beta * pen
        loss.backward(); opt.step()
        lv = loss.item()
        if lv < best - 1e-5: best, noimp = lv, 0
        else: noimp += 1
        if noimp > 300: break
    wall = time.perf_counter() - t0
    model.eval()
    q_mc = L.quotients(model, n=200_000)
    q_gr = L.quotients(model, grid=300)
    # overlap with previous (fixed grid)
    g = (torch.arange(300) + 0.5) / 300
    X = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        uk = model(X)
        ov = [float(((uk * mj(X)).mean() / ((uk ** 2).mean() * (mj(X) ** 2).mean()).sqrt()).abs()) for mj in models]
    E = q_gr["rq_grad"]
    nearest = min(exact, key=lambda e: abs(e - E))
    rec = dict(k=k + 1, E_grid=E, E_mc=q_mc["rq_grad"], E_lap_grid=q_gr["rq_lap"], exact_k=exact[k], nearest=nearest,
               rel_err_vs_kth=abs(E - exact[k]) / exact[k], rel_err_vs_nearest=abs(E - nearest) / nearest,
               epochs_run=ep, wall_time=wall, ms_per_epoch=1000 * wall / ep, final_loss=lv, max_overlap=max(ov, default=0.0),
               mean_u2=q_gr["mean_u2"])
    data["records"].append(rec)
    torch.save(model.state_dict(), L.CK / f"{tag}_{k + 1}.pt")
    L.dump_json(path, data)
    models.append(model)
    print(f"[{tag}] k={k+1} E={E:.4f} exact={exact[k]:.4f} relerr={rec['rel_err_vs_kth']:.2e} ep={ep} t={wall:.0f}s ov={rec['max_overlap']:.2e}", flush=True)
data["total_wall"] = sum(r["wall_time"] for r in data["records"])
L.dump_json(path, data)
print("total wall", data["total_wall"])
