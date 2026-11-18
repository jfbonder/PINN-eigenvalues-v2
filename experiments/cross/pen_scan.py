"""Warm-started scan with weak Dirichlet BC. usage: python -m experiments.cross.pen_scan --domain L --beta 100 --E0 6.4 --E1 24.4 --step 1.0 --mu 200 --decay 0.999"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from experiments.cross import cross_lib as C
from experiments.cross import pen as P
RES = Path(__file__).resolve().parent.parent / "results"
ap = argparse.ArgumentParser(); ap.add_argument("--domain", default="L"); ap.add_argument("--beta", type=float, default=100.0)
ap.add_argument("--E0", type=float, default=6.4); ap.add_argument("--E1", type=float, default=24.4); ap.add_argument("--step", type=float, default=1.0)
ap.add_argument("--mu", type=float, default=200.0); ap.add_argument("--decay", type=float, default=0.999); ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=""); a = ap.parse_args()
torch.set_num_threads(1)
samp = C.sample_L if a.domain == "L" else C.sample_cross; bs, tot = P.bnd_sampler(a.domain)
name = f"cross_scan_{a.domain}_b{a.beta:g}{a.tag}_s{a.seed}"; path = RES / f"{name}.json"; (RES / "cross_models").mkdir(exist_ok=True)
data = json.load(open(path)) if path.exists() else dict(args=vars(a), records=[])
Es = [round(a.E0 + i * a.step, 4) for i in range(int(round((a.E1 - a.E0) / a.step)) + 1)]
torch.manual_seed(a.seed); m = P.make(a.domain)
done = {r["E"] for r in data["records"]}
if data["records"]:
    m.load_state_dict(torch.load(RES / "cross_models" / f"{name}_last.pt"))
for E in Es:
    if E in done: continue
    H, w = P.train(m, E, samp, bs, a.mu, a.beta, epochs=a.epochs, decay=a.decay, hist=True)
    ev = P.evaluate(m, samp, bs, tot, E)
    ev.pop("chunk_r"); data["records"].append(dict(E=E, wall=w, hist_every10=[H[i] for i in range(0, len(H), 10)], final_loss=H[-1], **ev))
    json.dump(data, open(path, "w")); torch.save(m.state_dict(), RES / "cross_models" / f"{name}_last.pt")
    print(f"{a.domain} b={a.beta:g} E={E:6.2f} rho_res={ev['rho_res']:.4f} rho_var={ev['rho_var']:.4f} r={ev['r']:.3f} mass={ev['mass']:.3f} bdry_rel={ev['bdry_rel']:.1e} res={H[-1][1]:.3f} t={w:.0f}s", flush=True)
