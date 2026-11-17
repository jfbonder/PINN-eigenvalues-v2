"""Shared helpers for experiments/mu_cost (unit square, V=0, Dirichlet, vol=1)."""
from __future__ import annotations
import json, os, time, math
from functools import partial
from pathlib import Path
import numpy as np
import torch
from experiments import common as C
import pinn_eigen as pe

HERE = Path(__file__).resolve().parent
RES = C.RESULTS_DIR
CK = Path("/tmp/claude-0/-home-user-PINN-eigenvalues-v2/9cfb4ad8-ae89-5e57-bc88-c5e9420250f6/scratchpad/ckpt")
CK.mkdir(parents=True, exist_ok=True)

sampler = partial(pe.sample_square, x0=0.0, x1=1.0)
V = pe.V_zero
PI2 = math.pi ** 2


def exact_eigs(K=40):
    v = sorted(PI2 * (m * m + n * n) for m in range(1, 12) for n in range(1, 12))
    return v[:K]


def dist_to_spec(E):
    return min(abs(E - e) for e in exact_eigs(60))


def make_model(hidden=128, n_hidden=3):
    return pe.EigenFunctionNN(dim=2, boundary_fn=pe.boundary_square, hidden_size=hidden, n_hidden=n_hidden)


def lap_u(model, x):
    x = x.clone().requires_grad_(True)
    u = model(x)
    return u, pe._laplacian(u, x), x


def quotients(model, n=200_000, chunk=20000, grid=None):
    """Closed-form quotients on a FIXED sample (seeded generator) or midpoint grid.
    Returns dict: rq_lap = -sum(u Lu)/sum(u^2), rq_grad = sum|grad u|^2/sum u^2, mean_u2."""
    if grid:
        g = (torch.arange(grid) + 0.5) / grid
        X = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        n = X.shape[0]
    else:
        gen = torch.Generator().manual_seed(12345)
        X = torch.rand(n, 2, generator=gen)
    suL = su2 = sg = 0.0
    for i in range(0, n, chunk):
        x = X[i:i + chunk].clone().requires_grad_(True)
        u = model(x)
        (gu,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        lap = sum(torch.autograd.grad(gu[:, j:j + 1], x, torch.ones_like(u), create_graph=True)[0][:, j:j + 1]
                  for j in range(2))
        suL += (u * lap).detach().sum().item()
        su2 += (u ** 2).detach().sum().item()
        sg += (gu ** 2).detach().sum().item()
    return dict(rq_lap=-suL / su2, rq_grad=sg / su2, mean_u2=su2 / n)


def fresh_eval(model, E, n=20000):
    return pe.evaluate_residual(model, sampler, E, V, n_samples=n, vol=1.0)


def refine_old(model, E_lo, E_hi, n=10_000):
    """Existing pe.refine() logic: noisy re-sampled objective, bounded Brent."""
    from scipy.optimize import minimize_scalar
    def obj(E):
        x = sampler(n)
        _, res, _ = pe.loss_fn(model, x, E, V, mu=0.0, vol=1.0)
        return res
    r = minimize_scalar(obj, bounds=(E_lo, E_hi), method="bounded")
    return float(r.x), float(r.fun)


def parab_vertex(Es, ys):
    """3-point parabolic fit; Es equispaced (3), ys values. Returns vertex."""
    h = Es[1] - Es[0]
    y0, y1, y2 = ys
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-300:
        return float("nan")
    return float(Es[1] + 0.5 * h * (y0 - y2) / den)


def load_json(path, default):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return default


def dump_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=C._jsonable)
    os.replace(tmp, path)


def train_point(model, E, mu, n_epochs=3000, n_samples=5000, lr=1e-3, patience=300, tol=1e-5):
    """Train at fixed E (paper settings). Returns record (training terms + fresh eval)."""
    t0 = time.perf_counter()
    res, norm, info = pe.train_one(model, E, sampler, V, n_samples=n_samples, n_epochs=n_epochs, lr=lr,
                                   mu=mu, patience=patience, tol=tol, vol=1.0, return_info=True)
    ev = fresh_eval(model, E)
    rec = dict(E=float(E), mu=float(mu), train_loss_res=res, train_norm_err=norm,
               wall_time=time.perf_counter() - t0, epochs_run=info["epochs_run"],
               total_loss=info["total_loss"], eval_loss_res=ev["loss_res"], mass=ev["mass"], r=ev["r"],
               eval_norm_err=ev["norm_err"])
    return rec


def run_chain(tag, E_list, mu, seed, hidden=128, n_epochs=3000, n_samples=5000, patience=300,
              reset_every=50, ckdir=None, store=None, extra_quot=True):
    """Resumable warm-start chain over E_list. Results appended to RES/<tag>.json (list under 'records')."""
    path = RES / f"{tag}.json"
    data = load_json(path, dict(tag=tag, config=dict(mu=mu, seed=seed, hidden=hidden, n_hidden=3,
                              n_epochs=n_epochs, n_samples=n_samples, patience=patience, tol=1e-5, lr=1e-3,
                              reset_every=reset_every, E_list=list(map(float, E_list))), records=[]))
    done = {round(r["E"], 6) for r in data["records"]}
    torch.manual_seed(seed * 1000 + 7)
    model = make_model(hidden)
    # restore warm-start state from last checkpoint
    ck = CK / f"{tag}.pt"
    if data["records"] and ck.exists():
        model.load_state_dict(torch.load(ck))
    for i, E in enumerate(E_list):
        if round(float(E), 6) in done:
            continue
        if i > 0 and i % reset_every == 0:
            model = make_model(hidden)
        rec = train_point(model, float(E), mu if not callable(mu) else mu(E), n_epochs, n_samples, patience=patience)
        if extra_quot:
            q = quotients(model, n=100_000)
            rec.update(q)
        data["records"].append(rec)
        torch.save(model.state_dict(), ck)
        dump_json(path, data)
        print(f"[{tag}] E={E:.3f} mu={rec['mu']:g} res={rec['eval_loss_res']:.3e} mass={rec['mass']:.3f} "
              f"r={rec['r']:.3e} ep={rec['epochs_run']} t={rec['wall_time']:.0f}s", flush=True)
    return data
