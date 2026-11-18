"""Training diagnostics at single nodes: per-epoch loss histories + MC evaluation spread for training variants.
usage: python -m experiments.cross.diag VARIANT [VARIANT...] --E 26.1 --domain cross"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from functools import partial
from experiments.cross import cross_lib as C
RES = Path(__file__).resolve().parent.parent / "results"

VARIANTS = {
 "base":      dict(),
 "fixed":     dict(fixed_sample=True),
 "decay":     dict(lr_decay=0.998),
 "n20000":    dict(n_samples=20000),
 "fixed_decay": dict(fixed_sample=True, lr_decay=0.998),
 "mu3000":    dict(mu=3000.0),
 "eps":       dict(beps=0.02),
 "enr":       dict(enrich=True),
 "enr_eps":   dict(enrich=True, beps=0.02),
 "enr_eps_lr3": dict(enrich=True, beps=0.02, lr=3e-3),
 "enr_eps_mu30": dict(enrich=True, beps=0.02, mu=30.0),
 "enrB1":  dict(enrich=True, bpow=1, kmax=2),
 "enrB1_lr3": dict(enrich=True, bpow=1, kmax=2, lr=3e-3),
 "enrB1_k1": dict(enrich=True, bpow=1, kmax=1),
 "enr_eps_fixed": dict(enrich=True, beps=0.02, fixed_sample=True),
 "enr_eps_decay": dict(enrich=True, beps=0.02, lr_decay=0.998),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("variants", nargs="+"); ap.add_argument("--E", type=float, default=26.1)
    ap.add_argument("--domain", default="cross"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=2000)
    a = ap.parse_args(); torch.set_num_threads(1)
    samp = C.sample_cross if a.domain == "cross" else C.sample_L
    for v in a.variants:
        cfg = dict(mu=300.0, n_samples=5000); cfg.update(VARIANTS[v]); beps = cfg.pop("beps", 0.0); enrich = cfg.pop("enrich", False); bpow = cfg.pop("bpow", 0); kmax = cfg.pop("kmax", 1)
        out = RES / f"cross_diag_{a.domain}_E{a.E}_{v}_s{a.seed}.json"
        if out.exists(): continue
        B = (partial(C.B_cross, eps=beps, power=bpow) if a.domain == "cross" else partial(C.B_L, eps=beps, power=bpow))
        torch.manual_seed(a.seed); m = C.make_model(B, corners=(C.CORNERS_CROSS if a.domain == "cross" else C.CORNERS_L), enrich=enrich, kmax=kmax, R=(0.2 if a.domain == "cross" else 0.5))
        tr = C.train(m, a.E, samp, n_epochs=a.epochs, patience=10**9, **cfg)
        ev = C.evaluate(m, samp, a.E)
        torch.save(m.state_dict(), RES / "cross_models" / f"{a.domain}_E{a.E}_{v}_s{a.seed}.pt") if (RES / "cross_models").mkdir(exist_ok=True) is None else None
        json.dump(dict(variant=v, E=a.E, domain=a.domain, cfg=cfg, beps=beps, enrich=enrich, bpow=bpow, kmax=kmax, hist=tr["hist"], wall=tr["wall"], eval=ev), open(out, "w"))
        print(v, a.E, f"rho={ev['rho']:.4f}+-{ev['rho_se']:.4f} r={ev['r']:.3f}+-{ev['r_se']:.3f} mass={ev['mass']:.3f} chunk_r range [{min(ev['chunk_r']):.3f},{max(ev['chunk_r']):.3f}] wall={tr['wall']:.0f}s", flush=True)

if __name__ == "__main__":
    main()
