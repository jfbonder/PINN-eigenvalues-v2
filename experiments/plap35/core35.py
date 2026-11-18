"""p-Laplacian, p = 3.5, unit disk: baseline strong-form loss and remedies (exploratory; pinn_eigen.py untouched).

Variants (job["variant"]):
  base     : int (Delta_p u + E w)^2 + mu (int|u|^p - 1)^2,   w = |u|^{p-2}u                      [current method]
  mixed    : network outputs (u, sigma);  int (div sigma + E w)^2 + lam int |sigma - |grad u|^{p-2} grad u|^2 + mu(...)^2
  wdown    : int omega (Delta_p u + E w)^2 + ...,  omega = |grad u|^2/(|grad u|^2 + kappa^2)   (down-weights critical points)
  wdiv     : int [(Delta_p u + E w)/(|grad u|^2+kappa^2)^{(p-2)/2}]^2 + ...              (literal "divide by the degenerate factor")
  + options: N (MC points), imp (importance sampling: fraction of points drawn uniformly in the disk of radius imp_r around 0,
             with the exact importance weights), mu, epochs, seed.
Protocol P1 everywhere: Adam, lr 1e-3 -> 1e-5 cosine, fixed number of epochs per node, warm start along the grid.

Per-node record (fresh UNIFORM sample of n_eval points, integrals = pi * mean):
  loss_var  : value of the variant's E-dependent functional (res term [+ lam * constraint term]) / mass^{2(p-1)/p}
  Lam_strong: int (Delta_p u + E w)^2 / mass^{2(p-1)/p}       (strong residual of u, comparable across variants)
  E_var     : closed-form minimiser in E of the variant functional for the frozen network  ("E_PINN" of the variant)
  E_strong  : -<Delta_p u, w>/<w, w>   (baseline refinement applied to u)
  E_weak    : mixed only:  int sigma.grad u / int |u|^p       RQ: int|grad u|^p / int|u|^p (MC error ~0.5-1 %)
  cons      : mixed only: lam-free constraint term int |sigma - flux|^2
  core_frac : share of the variant residual carried by the 2 % of points with smallest |grad u|;  mass = int |u|^p
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
CKPT = HERE / "ckpt"
VOL = float(np.pi)


class Net(nn.Module):
    def __init__(self, n_out, width=128, depth=3):
        super().__init__()
        L = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            L += [nn.Linear(width, width), nn.Tanh()]
        L.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*L)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh")); nn.init.zeros_(m.bias)

    def forward(self, x):
        o = self.net(x)
        B = 1.0 - (x ** 2).sum(1, keepdim=True)
        return B * o[:, 0:1], o[:, 1:]            # u (Dirichlet), sigma (free; empty unless mixed)


def make_model(job):
    return Net(3 if job["variant"] == "mixed" else 1, job.get("width", 128), job.get("depth", 3))


def sample(n, job):
    """Uniform points in the unit disk, or the importance-sampling mixture; returns (x, weights with E_q[w]=1)."""
    f = float(job.get("imp", 0.0))
    r = torch.sqrt(torch.rand(n, 1)); th = 2 * np.pi * torch.rand(n, 1)
    if f > 0:
        r0 = float(job.get("imp_r", 0.3)); k = int(f * n)
        r[:k] *= r0
        dens_ratio = (1 - f) + f / r0 ** 2 * (r < r0).float()      # q(x) / uniform density
        wq = 1.0 / dens_ratio
    else:
        wq = torch.ones(n, 1)
    return torch.cat([r * torch.cos(th), r * torch.sin(th)], 1), wq


def div(F, x, create_graph):
    return sum(torch.autograd.grad(F[:, i].sum(), x, create_graph=create_graph, retain_graph=True)[0][:, i:i + 1]
               for i in range(2))


def terms(model, x, job, need_strong, create_graph):
    """Pointwise quantities: a (E-independent part of the variant residual), b (coefficient of E), aux (constraint), ..."""
    p = float(job["p"]); v = job["variant"]
    x = x.requires_grad_(True)
    u, sig = model(x)
    (g,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
    g2 = g.pow(2).sum(1, keepdim=True)
    flux = (g2 + 1e-24) ** ((p - 2) / 2) * g
    w = u.abs().clamp_min(1e-12) ** (p - 2) * u
    out = dict(u=u, w=w, g2=g2, dens=u.abs() ** p)
    if need_strong or v != "mixed":
        out["Lu"] = div(flux, x, create_graph)
    if v == "mixed":
        out["a"] = div(sig, x, create_graph); out["b"] = w
        out["aux"] = (sig - flux).pow(2).sum(1, keepdim=True); out["sig_gradu"] = (sig * g).sum(1, keepdim=True)
    else:
        if v == "wdown":
            om = torch.sqrt(g2 / (g2 + float(job["kappa"]) ** 2))
        elif v == "wdiv":
            om = (g2 + float(job["kappa"]) ** 2) ** (-(p - 2) / 2)
        else:
            om = torch.ones_like(u)
        out["a"] = om * out["Lu"]; out["b"] = om * w
    return out


def train_node(model, E, job, epochs):
    p = float(job["p"]); mu = float(job["mu"]); lam = float(job.get("lam", 0.0)); N = int(job.get("N", 5000))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    t0 = time.perf_counter()
    for _ in range(epochs):
        x, wq = sample(N, job)
        opt.zero_grad()
        T = terms(model, x, job, False, True)
        res = VOL * (wq * (T["a"] + E * T["b"]) ** 2).mean()
        nrm = (VOL * (wq * T["dens"]).mean() - 1.0) ** 2
        loss = res + mu * nrm
        if job["variant"] == "mixed":
            loss = loss + lam * VOL * (wq * T["aux"]).mean()
        loss.backward(); opt.step(); sch.step()
    return dict(train_res=res.item(), train_norm=nrm.item(), train_loss=loss.item(), wall_time=time.perf_counter() - t0, epochs_run=epochs)


def evaluate(model, E, job, n_eval=40000, chunk=5000):
    p = float(job["p"]); lam = float(job.get("lam", 0.0)); mixed = job["variant"] == "mixed"
    keys = ["a", "b", "Lu", "w", "g2", "dens"] + (["aux", "sig_gradu"] if mixed else [])
    acc = {k: [] for k in keys}
    for _ in range(n_eval // chunk):
        x, _ = sample(chunk, dict(job, imp=0.0))
        T = terms(model, x, job, True, False)
        for k in keys:
            acc[k].append(T[k].detach().flatten())
    A = {k: torch.cat(v).double() for k, v in acc.items()}
    m = lambda t: t.mean().item()
    mass = VOL * m(A["dens"]); hom = mass ** (2 * (p - 1) / p)
    E_var = -m(A["a"] * A["b"]) / m(A["b"] ** 2)
    E_strong = -m(A["Lu"] * A["w"]) / m(A["w"] ** 2)
    rv = (A["a"] + E * A["b"]) ** 2
    rs = (A["Lu"] + E * A["w"]) ** 2
    k = max(1, int(0.02 * len(rv))); idx = torch.topk(A["g2"], k, largest=False).indices
    out = dict(mass=mass, res_var=VOL * m(rv) / hom, Lam_strong=VOL * m(rs) / hom, E_var=E_var, E_strong=E_strong,
               RQ=m(A["g2"] ** (p / 2)) / m(A["dens"]), core_frac=(rv[idx].sum() / rv.sum()).item(),
               core_frac_strong=(rs[idx].sum() / rs.sum()).item(), collapsed=bool(mass < 0.5))
    out["loss_var"] = out["res_var"]
    if mixed:
        out["cons"] = VOL * m(A["aux"]) / hom
        out["loss_var"] = out["res_var"] + lam * out["cons"]
        out["E_weak"] = m(A["sig_gradu"]) / m(A["dens"])
    return out


def run_job(job):
    name = job["name"]; path = RES / f"plap35_{name}.json"; ck = CKPT / f"{name}.pt"
    Es = [float(e) for e in job["E"]]; recs = []
    if path.exists():
        old = json.load(open(path))
        if old["config"]["E"] == Es and ck.exists():
            recs = old["records"]
    seed = int(job.get("seed", 0)); torch.manual_seed(seed)
    model = make_model(job)
    if recs:
        model.load_state_dict(torch.load(ck))
    for i, E in enumerate(Es):
        if i < len(recs):
            continue
        torch.manual_seed(seed * 100003 + i)
        ep = job.get("epochs0", job["epochs"]) if i == 0 else job["epochs"]
        info = train_node(model, E, job, ep)
        ev = evaluate(model, E, job)
        recs.append(dict(E=E, **info, **ev))
        torch.save(model.state_dict(), ck)
        tmp = str(path) + ".tmp"
        json.dump(dict(name=name, config=dict(job, E=Es), records=recs, total_wall_s=sum(r["wall_time"] for r in recs),
                       host=dict(threads=torch.get_num_threads(), torch=torch.__version__)), open(tmp, "w"), indent=1)
        os.replace(tmp, path)
        print(f"[{name}] E={E:7.3f} loss_var={ev['loss_var']:.3e} Lam_strong={ev['Lam_strong']:.3e} mass={ev['mass']:.3f} "
              f"E_var={ev['E_var']:.4f} E_strong={ev['E_strong']:.4f}" + (f" E_weak={ev['E_weak']:.4f} cons={ev['cons']:.2e}" if "cons" in ev else "")
              + f" core={ev['core_frac']:.2f} t={info['wall_time']:.0f}s", flush=True)
