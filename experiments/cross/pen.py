"""Weak (penalty) Dirichlet BC: u = plain MLP, loss = mean(res^2) + mu (mean u^2 - 1)^2 + beta * mean_{boundary pts}(u^2).
usage: python -m experiments.cross.pen --domain L --E 9.64 --betas 100 1000 --tag x [--decay 0.999] [--epochs 2000]"""
import argparse, json, time, math
from pathlib import Path
import numpy as np, torch
from experiments.cross import cross_lib as C
import pinn_eigen as pe
RES = Path(__file__).resolve().parent.parent / "results"

def bnd_sampler(domain):
    if domain == "L":
        segs = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 0), (1, -1)), ((0, 1), (-1, 1)), ((-1, 1), (-1, -1)), ((-1, -1), (1, -1))]
    else:
        a = C.A
        pts = [(1, a), (a, a), (a, 1), (-a, 1), (-a, a), (-1, a), (-1, -a), (-a, -a), (-a, -1), (a, -1), (a, -a), (1, -a)]
        segs = [(pts[i], pts[(i + 1) % 12]) for i in range(12)]
    P = torch.tensor(segs, dtype=torch.float32); L = (P[:, 1] - P[:, 0]).norm(dim=1); tot = float(L.sum())
    def sampler(n):
        k = torch.multinomial(L / L.sum(), n, replacement=True); t = torch.rand(n, 1)
        return P[k, 0] + t * (P[k, 1] - P[k, 0])
    return sampler, tot

def make(domain):
    one = lambda x: torch.ones(x.shape[0], 1)
    return C.Enriched(one, [], enrich=False)

def evaluate(m, samp, bs, tot, E, n_chunks=20, chunk=10000):
    st = torch.get_rng_state(); torch.manual_seed(777); S = []
    for _ in range(n_chunks):
        x = samp(chunk).requires_grad_(True); u = m.core(x); (g,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        lap = pe._laplacian(u, x); res = lap + E * u
        S.append([(u ** 2).mean().item(), (u * lap).mean().item(), (g ** 2).sum(1).mean().item(), (res ** 2).mean().item()])
    xb = bs(20000); ub = m.core(xb); bmean = (ub ** 2).mean().item(); bmax = ub.abs().max().item()
    torch.set_rng_state(st)
    S = np.array(S); u2, ulu, g2, r2 = S.T; m_ = S.mean(0)
    se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a)))
    return dict(rho_res=float(-m_[1] / m_[0]), rho_res_se=se(-ulu / u2), rho_var=float(m_[2] / m_[0]), rho_var_se=se(g2 / u2), r=float(np.sqrt(m_[3] / m_[0])),
                r_se=se(np.sqrt(r2 / u2)), mass=float(m_[0]), bdry_int_u2=bmean * tot, bdry_mean_u2=bmean, bdry_max_u=bmax,
                bdry_rel=float(bmean / m_[0]), chunk_r=np.sqrt(r2 / u2).tolist())

def train(m, E, samp, bs, mu, beta, n=5000, nb=1500, epochs=2000, lr=1e-3, decay=None, patience=10**9, hist=True):
    opt = torch.optim.Adam(m.parameters(), lr=lr); sch = torch.optim.lr_scheduler.ExponentialLR(opt, decay) if decay else None
    H = []; t0 = time.perf_counter()
    for ep in range(epochs):
        x = samp(n); xb = bs(nb); opt.zero_grad()
        u, rs, _ = m.u_res(x, E); res = (rs ** 2).mean(); nm = (u ** 2).mean() - 1
        ub = m.core(xb); bl = (ub ** 2).mean()
        loss = res + mu * nm ** 2 + beta * bl; loss.backward(); opt.step()
        if sch: sch.step()
        if hist: H.append((loss.item(), res.item(), nm.item() ** 2, bl.item()))
    return H, time.perf_counter() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", default="L"); ap.add_argument("--E", type=float, default=9.64)
    ap.add_argument("--betas", type=float, nargs="+", default=[100, 1000]); ap.add_argument("--mu", type=float, default=100.0)
    ap.add_argument("--decay", type=float, default=None); ap.add_argument("--epochs", type=int, default=2000); ap.add_argument("--tag", default="")
    ap.add_argument("--nb", type=int, default=1500); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    torch.set_num_threads(1)
    samp = C.sample_L if a.domain == "L" else C.sample_cross; bs, tot = bnd_sampler(a.domain)
    for beta in a.betas:
        out = RES / f"cross_pen_{a.domain}_E{a.E}_b{beta:g}{a.tag}_s{a.seed}.json"
        if out.exists(): continue
        torch.manual_seed(a.seed); m = make(a.domain)
        H, w = train(m, a.E, samp, bs, a.mu, beta, nb=a.nb, epochs=a.epochs, decay=a.decay)
        ev = evaluate(m, samp, bs, tot, a.E); (RES / "cross_models").mkdir(exist_ok=True)
        torch.save(m.state_dict(), RES / "cross_models" / f"pen_{a.domain}_E{a.E}_b{beta:g}{a.tag}_s{a.seed}.pt")
        json.dump(dict(domain=a.domain, E=a.E, beta=beta, mu=a.mu, decay=a.decay, nb=a.nb, hist=H, wall=w, eval=ev, bdry_len=tot), open(out, "w"))
        print(f"{a.domain} E={a.E} beta={beta:g}{a.tag}: rho_res={ev['rho_res']:.4f} rho_var={ev['rho_var']:.4f}+-{ev['rho_var_se']:.4f} r={ev['r']:.3f} mass={ev['mass']:.3f} "
              f"bdry_int_u2={ev['bdry_int_u2']:.2e} (rel {ev['bdry_rel']:.1e}) wall={w:.0f}s", flush=True)
