"""Graded Monte Carlo for the L-shape (weak Dirichlet BC, penalty). Variants: 0 uniform, W weighted (unbiased), U unweighted graded mixture.
usage: python -m experiments.cross.graded --k 1 --variant W --seed 0 [--epochs 1000] [--beta 300]
Warm start from the _ck scan net nearest the minimum, local scan step 0.1. Evaluation always uniform (pen.evaluate)."""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch
from experiments.cross import cross_lib as C
from experiments.cross import pen as P
RES = Path(__file__).resolve().parent.parent / "results"
TH = 1.5 * math.pi

def graded_interior(n, alpha, k, R=1.0):
    """returns x (n,2), q (n,) mixture density w.r.t. Lebesgue measure on the L (uniform density = 1/3)."""
    ng = int(round(alpha * n)); xu = C.sample_L(n - ng)
    s = torch.rand(ng); r = R * s ** k; th = math.pi / 2 + TH * torch.rand(ng)
    xg = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    x = torch.cat([xu, xg], 0)
    return x, mix_density(x, alpha, k, R)

def mix_density(x, alpha, k, R=1.0):
    r = x.norm(dim=1).clamp_min(1e-12)
    pg = torch.where(r < R, (1.0 / TH) * (1.0 / k) * R ** (-1.0 / k) * r ** (1.0 / k - 2.0), torch.zeros_like(r))
    return (1 - alpha) / 3.0 + alpha * pg

def graded_boundary(n, alpha, k, bs_uniform, tot=8.0):
    """alpha of the points on the two re-entrant edges graded towards the corner; density w.r.t. arclength."""
    ng = int(round(alpha * n)); xu = bs_uniform(n - ng)
    s = torch.rand(ng); ax = torch.randint(0, 2, (ng,)).bool(); t = s ** k
    xg = torch.where(ax[:, None], torch.stack([t, 0 * t], 1), torch.stack([0 * t, t], 1))
    x = torch.cat([xu, xg], 0)
    on = ((x[:, 1] == 0) & (x[:, 0] >= 0) & (x[:, 0] <= 1)) | ((x[:, 0] == 0) & (x[:, 1] >= 0) & (x[:, 1] <= 1))
    d = torch.where(x[:, 1] == 0, x[:, 0], x[:, 1]).clamp_min(1e-12)   # distance from corner along the edge
    pg = 0.5 * (1.0 / k) * d ** (1.0 / k - 1.0)
    return x, (1 - alpha) / tot + alpha * pg * on.float()

def train_graded(m, E, variant, alpha, k, bs, mu, beta, n=5000, nb=1500, epochs=1000, lr=5e-4, decay=0.998):
    opt = torch.optim.Adam(m.parameters(), lr=lr); sch = torch.optim.lr_scheduler.ExponentialLR(opt, decay); H = []; t0 = time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad()
        if variant == "0":
            x = C.sample_L(n); xb = bs(nb); w = wb = None
        else:
            x, q = graded_interior(n, alpha, k); xb, qb = graded_boundary(nb, alpha, k, bs)
            if variant == "W": w = (1 / 3.0) / q; wb = (1 / 8.0) / qb
            else: w = wb = None
        u, rs, _ = m.u_res(x, E); ub = m.core(xb)
        wm = (lambda f, w: f.mean()) if w is None else (lambda f, w: (w.view_as(f) * f).sum() / w.sum())
        res = wm(rs ** 2, w); nm = wm(u ** 2, w) - 1; bl = wm(ub ** 2, wb)
        loss = res + mu * nm ** 2 + beta * bl; loss.backward(); opt.step(); sch.step()
        H.append((loss.item(), res.item(), nm.item() ** 2, bl.item()))
    return H, time.perf_counter() - t0

def corner_share(m, E, rc=0.2, n=200000, seed=4242):
    """share of sum(res^2) within rc of the corner, uniform points."""
    st = torch.get_rng_state(); torch.manual_seed(seed); tot = 0.0; near = 0.0
    for _ in range(n // 20000):
        x = C.sample_L(20000); _, rs, _ = m.u_res(x, E); r2 = (rs ** 2).detach().squeeze(); c = x.norm(dim=1) < rc
        tot += r2.sum().item(); near += r2[c].sum().item()
    torch.set_rng_state(st); return near / tot

GRIDS = {1: (9.2, 10.1, 9.4), 2: (14.8, 15.6, 15.4), 3: (19.3, 20.2, 19.4)}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=1); ap.add_argument("--variant", default="W"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1000); ap.add_argument("--beta", type=float, default=300.0); ap.add_argument("--mu", type=float, default=200.0)
    ap.add_argument("--alpha", type=float, default=0.25); ap.add_argument("--power", type=float, default=2.5); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--decay", type=float, default=0.998); ap.add_argument("--init", default=None, help="checkpoint (default: _ck node)"); ap.add_argument("--tag", default="")
    ap.add_argument("--E0", type=float, default=None); ap.add_argument("--E1", type=float, default=None); a = ap.parse_args()
    torch.set_num_threads(1); bs, tot = P.bnd_sampler("L"); lo, hi, start = GRIDS[a.k]
    lo = a.E0 if a.E0 is not None else lo; hi = a.E1 if a.E1 is not None else hi
    Es = [round(lo + 0.1 * i, 4) for i in range(int(round((hi - lo) / 0.1)) + 1)]
    name = f"cross_graded_L_k{a.k}_{a.variant}_s{a.seed}_b{a.beta:g}{a.tag}"; out = RES / f"{name}.json"; (RES / "cross_models").mkdir(exist_ok=True)
    init = a.init or str(RES / "cross_models" / f"cross_scan_L_b300_ck_s0_E{start}.pt")
    torch.manual_seed(1000 + a.seed); m = P.make("L"); m.load_state_dict(torch.load(init))
    data = dict(args=vars(a), init=init, records=[]); ev0 = P.evaluate(m, C.sample_L, bs, tot, start); ev0.pop("chunk_r"); data["start_eval"] = ev0
    # order: nodes below the start descending would break warm start monotonicity; simply ascend from lo
    for E in Es:
        H, w = train_graded(m, E, a.variant, a.alpha, a.power, bs, a.mu, a.beta, epochs=a.epochs, lr=a.lr, decay=a.decay)
        ev = P.evaluate(m, C.sample_L, bs, tot, E); ev.pop("chunk_r"); cs = corner_share(m, E)
        torch.save(m.state_dict(), RES / "cross_models" / f"{name}_E{E:g}.pt")
        data["records"].append(dict(E=E, wall=w, final_loss=H[-1], hist_every20=[H[i] for i in range(0, len(H), 20)], corner_share=cs, **ev))
        json.dump(data, open(out, "w"))
        print(f"k{a.k} {a.variant} s{a.seed} E={E:5.2f} rho_res={ev['rho_res']:.4f} rho_var={ev['rho_var']:.4f} r={ev['r']:.3f} mass={ev['mass']:.3f} bdry={ev['bdry_rel']:.1e} corner={cs:.2f} t={w:.0f}s", flush=True)
