"""Core helpers for the p-Laplacian experiments (agent E).  Uses pinn_eigen.py unchanged.

A *job* is a dict (see jobs.py) describing one scan in E: domain, p, E grid, network, MC size, epochs, mu, seed,
legacy flag.  run_job() trains with pe.train_one (warm start along the grid), evaluates every trained network on a
fresh sample and appends one record per E to experiments/results/plap_<name>.json (resumable: finished E are
skipped, the last network is checkpointed in experiments/plap/ckpt/).

Per-E record (all integrals are |Omega| * MC mean on a FRESH sample of n_eval points):
  train_res, train_norm : last-epoch training-batch values (what the paper's tables called "Residual")
  Lam      : int (Delta_p u + E|u|^{p-2}u)^2            (fresh)                [legacy: mean((Delta_p u + E u)^2)]
  mass     : int |u|^p                                                         [legacy: mean(u^2)]
  Lam_n    : Lam / mass^{2(p-1)/p}   scale-invariant residual (= Lam if mass = 1)
  E_ls     : argmin_E of the frozen-network residual, closed form  -<Delta_p u, w>/<w, w>, w=|u|^{p-2}u
  Lam_ls   : the frozen-network residual at E_ls (normalised like Lam_n)
  RQ       : Rayleigh quotient int|grad u|^p / int|u|^p of the trained network
  core_frac: fraction of Lam coming from the 2% of sample points with the smallest |grad u| (critical set)
"""
from __future__ import annotations
import json, os, time
from functools import partial
from pathlib import Path
import numpy as np
import torch

from experiments import common as C
import pinn_eigen as pe

HERE = Path(__file__).resolve().parent
RES = C.RESULTS_DIR
CKPT = HERE / "ckpt"

DOMAINS = {
    "disk":   dict(dim=2, sampler=partial(pe.sample_disk, radius=1.0), B=pe.boundary_disk, vol=pe.VOL_DISK),
    "square2": dict(dim=2, sampler=partial(pe.sample_square, x0=0.0, x1=2.0), B=pe.boundary_square_2x2,
                    vol=pe.VOL_SQUARE_2X2),
    "ball3":  dict(dim=3, sampler=pe.sample_ball_3d, B=pe.boundary_ball, vol=pe.VOL_BALL_3D),
}


def make_model(job):
    D = DOMAINS[job["domain"]]
    return pe.EigenFunctionNN(dim=D["dim"], boundary_fn=D["B"], hidden_size=job.get("width", 128),
                              n_hidden=job.get("depth", 3))


def evaluate(model, job, E, n_eval=40000, chunk=5000, seed=None):
    """Fresh-sample diagnostics (see module docstring)."""
    D = DOMAINS[job["domain"]]; p = float(job["p"]); legacy = bool(job.get("legacy", False))
    vol = 1.0 if legacy else D["vol"]
    if seed is not None:
        torch.manual_seed(seed)
    acc = dict(lu_w=0.0, w_w=0.0, lu_lu=0.0, dens=0.0, gp=0.0, up=0.0, u2=0.0)
    res_all, g_all = [], []
    n = 0
    while n < n_eval:
        x = D["sampler"](chunk).requires_grad_(True)
        u = model(x)
        (g,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        gn = torch.norm(g, dim=1, keepdim=True) + 1e-12
        if p == 2.0:
            Lu = pe._laplacian(u, x)
        else:
            A = gn ** (p - 2) * g
            Lu = sum(torch.autograd.grad(A[:, i].sum(), x, retain_graph=True)[0][:, i:i + 1]
                     for i in range(x.shape[1]))
        u = u.detach(); Lu = Lu.detach(); gn = gn.detach()
        w = u if (legacy or p == 2.0) else u.abs().clamp_min(1e-12) ** (p - 2.0) * u
        dens = u ** 2 if (legacy or p == 2.0) else u.abs() ** p
        acc["lu_w"] += (Lu * w).sum().item(); acc["w_w"] += (w * w).sum().item()
        acc["lu_lu"] += (Lu * Lu).sum().item(); acc["dens"] += dens.sum().item()
        acc["gp"] += (gn ** p).sum().item(); acc["up"] += (u.abs() ** p).sum().item()
        acc["u2"] += (u ** 2).sum().item()
        res_all.append(((Lu + E * w) ** 2).flatten()); g_all.append(gn.flatten())
        n += chunk
    res2 = torch.cat(res_all); gn = torch.cat(g_all)
    m = {k: v / n for k, v in acc.items()}
    Lam = vol * res2.mean().item()
    mass = vol * m["dens"]
    hom = 1.0 if (legacy or p == 2.0) else 2.0 * (p - 1.0) / p     # Lam ~ mass^hom under u -> t u
    E_ls = -m["lu_w"] / m["w_w"]
    Lam_ls = vol * (m["lu_lu"] - m["lu_w"] ** 2 / m["w_w"])
    k = max(1, int(0.02 * n))
    idx = torch.topk(gn, k, largest=False).indices
    return dict(Lam=Lam, mass=mass, Lam_n=Lam / mass ** hom, E_ls=E_ls, Lam_ls=Lam_ls / mass ** hom,
                RQ=m["gp"] / m["up"], core_frac=(res2[idx].sum() / res2.sum()).item(),
                res_max=res2.max().item() ** 0.5, mean_u2=m["u2"])


def _lr_sched(job):
    kind = job.get("lr_decay")
    if not kind:
        return None
    n = job["epochs"]
    return lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n, eta_min=job.get("lr", 1e-3) * 0.01)


def run_job(job, verbose=True):
    name = job["name"]
    path = RES / f"plap_{name}.json"
    ck = CKPT / f"{name}.pt"
    CKPT.mkdir(exist_ok=True)
    D = DOMAINS[job["domain"]]; p = float(job["p"]); legacy = bool(job.get("legacy", False))
    vol = 1.0 if legacy else D["vol"]
    Es = [float(e) for e in job["E"]]
    recs = []
    if path.exists():
        old = json.load(open(path))
        if old.get("config", {}).get("E") == Es:
            recs = old["records"]
    done = len(recs)
    seed = int(job.get("seed", 0))
    torch.manual_seed(seed)
    model = make_model(job)
    if done > 0 and ck.exists():
        model.load_state_dict(torch.load(ck))
    elif done > 0:
        recs, done = [], 0
    reset_every = job.get("reset_every", 10 ** 9)
    for i, E in enumerate(Es):
        if i < done:
            continue
        torch.manual_seed(seed * 100003 + i)          # deterministic also after a resume
        if i > 0 and i % reset_every == 0 or (i > 0 and not job.get("warm", True)):
            model = make_model(job)
        mu = job["mu"]
        epochs = job["epochs"] if (i % reset_every != 0 or "epochs0" not in job) else job["epochs0"]
        if i == 0 and "epochs0" in job:
            epochs = job["epochs0"]
        res, nrm, info = pe.train_one(model, E, D["sampler"], pe.V_zero, n_samples=job.get("N", 5000),
                                      n_epochs=epochs, lr=job.get("lr", 1e-3), mu=mu, p=p,
                                      patience=job.get("patience", 300), tol=job.get("tol", 1e-5),
                                      lr_scheduler=_lr_sched(dict(job, epochs=epochs)),
                                      legacy_plap=legacy, vol=vol, return_info=True)
        ev = evaluate(model, job, E, n_eval=job.get("n_eval", 40000))
        rec = dict(E=E, mu=mu, train_res=res, train_norm=nrm, **info, **ev)
        recs.append(rec)
        torch.save(model.state_dict(), ck)
        if job.get("keep_models"):
            torch.save(model.state_dict(), CKPT / f"{name}_E{E:.4f}.pt")
        payload = dict(name=name, config=dict(job, E=Es), records=recs,
                       host=dict(threads=torch.get_num_threads(), torch=torch.__version__),
                       total_wall_s=sum(r["wall_time"] for r in recs))
        tmp = str(path) + ".tmp"
        json.dump(payload, open(tmp, "w"), indent=1); os.replace(tmp, path)
        if verbose:
            print(f"[{name}] E={E:8.4f} Lam={ev['Lam']:.4e} mass={ev['mass']:.4f} Lam_n={ev['Lam_n']:.4e} "
                  f"E_ls={ev['E_ls']:.4f} RQ={ev['RQ']:.4f} core={ev['core_frac']:.2f} "
                  f"ep={info['epochs_run']} t={info['wall_time']:.0f}s", flush=True)
    return recs


def parabola_min(E, L):
    """Vertex of the parabola through the lowest grid point and its two neighbours (3-point fit)."""
    E = np.asarray(E, float); L = np.asarray(L, float)
    i = int(np.argmin(L)); i = min(max(i, 1), len(E) - 2)
    c = np.polyfit(E[i - 1:i + 2], L[i - 1:i + 2], 2)
    return float(-c[1] / (2 * c[0])), float(np.polyval(c, -c[1] / (2 * c[0])))
