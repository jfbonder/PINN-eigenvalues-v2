"""Extract PINN eigenvalue estimates from window sweeps and compare with FEM (experiments/results/fem_summary.json)."""
import json, glob
import numpy as np
from pathlib import Path
RES = Path(__file__).resolve().parent.parent / "results"

def parab(E, y, i):
    """3-point vertex around node i (theory_notes, prop.parabola)."""
    if i == 0 or i == len(E) - 1: return float("nan")
    h = E[i + 1] - E[i]
    den = y[i - 1] - 2 * y[i] + y[i + 1]
    if den <= 0: return float("nan")
    return float(E[i] + 0.5 * h * (y[i - 1] - y[i + 1]) / den)

def local_minima(E, y):
    return [i for i in range(1, len(E) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]

def analyse_window(w, fem_rows):
    recs = sorted(w["records"], key=lambda r: r["E"])
    E = np.array([r["E"] for r in recs])
    r2 = np.array([r["r2"] for r in recs]); lr = np.array([r["loss_res"] for r in recs])
    rv = np.array([r["rho_var"] for r in recs]); rr = np.array([r["rho_res"] for r in recs])
    if len(E) < 3: return None
    i2 = int(np.argmin(r2)); il = int(np.argmin(lr))
    out = dict(center=w["center"], n=len(E), E=E.tolist(), r2=r2.tolist(), loss_res=lr.tolist(),
               rho_var=rv.tolist(), rho_res=rr.tolist(),
               node_min_r2=float(E[i2]), node_min_loss=float(E[il]),
               E_par_r2=parab(E, r2, i2), E_par_loss=parab(E, lr, il),
               rho_var_at_min=float(rv[i2]), rho_res_at_min=float(rr[i2]),
               n_local_min_loss=len(local_minima(E, lr)), n_local_min_r2=len(local_minima(E, r2)),
               min_r2=float(r2[i2]), min_loss=float(lr[il]),
               time_total=float(sum(r["wall"] for r in recs)), epochs=[r["epochs_run"] for r in recs],
               time_per_E=float(np.mean([r["wall"] for r in recs])))
    lo, hi = E.min(), E.max()
    inside = [r for r in fem_rows if lo - 0.5 <= r["E"] <= hi + 0.5]
    out["fem_in_window"] = [dict(E=r["E"], mult=r["mult"], err=r["err"]) for r in inside]
    # reference = FEM eigenvalue (cluster) nearest to the PINN estimate (rho_var at min node)
    if inside:
        est = out["rho_var_at_min"]
        ref = min(inside, key=lambda r: abs(r["E"] - est))
        out["ref"] = dict(E=ref["E"], mult=ref["mult"], err=ref["err"])
    return out

def load_all(case, seeds=(0, 1, 2)):
    summ = json.load(open(RES / "fem_summary.json"))[case]["rows"]
    res = {}
    for s in seeds:
        p = RES / f"fem_pinn_{case}_s{s}.json"
        if not p.exists(): continue
        d = json.load(open(p))
        res[s] = dict(settings=d["settings"], windows={k: analyse_window(w, summ) for k, w in d["windows"].items()})
    return summ, res
