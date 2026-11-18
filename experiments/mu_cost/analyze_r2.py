"""Round 2 analysis: (a) offset-grid mu test on [15,55]; (b) cost; (c) cold start."""
import json, glob, numpy as np
from experiments.mu_cost import lib as L
EX = sorted(set(np.round(L.exact_eigs(60), 6)))
near = lambda x: min(EX, key=lambda e: abs(e - x))
dd = lambda E: abs(E - near(E))
Ef = np.arange(15, 55.0001, 0.01); D2 = max(dd(E) for E in Ef) ** 2
print("D^2 [15,55] =", D2)
def closed_res(E, mu):
    d2 = dd(E) ** 2; return d2 * max(0.0, 1 - d2 / (2 * mu))
def locmin(y): return [i for i in range(1, len(y) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]
rows = []
for mu in (30, 75, 150, 219, 300, 1000):
    for s in (0, 1):
        d = L.load_json(L.RES / f"mucost_off_mu{mu}_s{s}.json", None)
        if not d or len(d["records"]) < 16: continue
        R = d["records"]; E = np.array([r["E"] for r in R]); res = np.array([r["eval_loss_res"] for r in R])
        mass = np.array([r["mass"] for r in R]); r2 = np.array([r["r"] ** 2 for r in R])
        rho = np.array([r["rq_lap"] for r in R])
        d2 = np.array([dd(e) ** 2 for e in E])
        pm = np.array([max(0, 1 - x / (2 * mu)) for x in d2])
        # interior minima of the recorded residual and of normalised quotient
        mr = [float(E[i]) for i in locmin(res)]; mq = [float(E[i]) for i in locmin(r2)]
        # which are near an eigenvalue (|E-eig|<=2.5)
        sp_r = [e for e in mr if dd(e) > 2.5]; sp_q = [e for e in mq if dd(e) > 2.5]
        # distortion: recorded residual / predicted
        pred = np.array([closed_res(e, mu) for e in E])
        dist = float(np.max(np.abs(res - pred) / np.maximum(pred, 1.0)))
        coll = int((mass < 0.1).sum()); pcoll = int((pm < 0.1).sum())
        # eigenvalue estimates: RQ at nodes with smallest r2 near each eigenvalue
        est = {}
        for ek in (2 * L.PI2, 5 * L.PI2):
            idx = [i for i in range(len(E)) if abs(E[i] - ek) < 2.6 and mass[i] > 0.1]
            if idx:
                i = min(idx, key=lambda j: r2[j]); est[round(ek, 3)] = float(rho[i])
        rows.append(dict(mu=mu, seed=s, mu_over_D2=mu / D2, minima_rec=mr, minima_norm=mq, spurious_rec=sp_r, spurious_norm=sp_q,
                         collapsed=coll, pred_collapsed=pcoll, max_rel_distortion=dist, mass_rms=float(np.sqrt(np.mean((mass - pm) ** 2))), est=est))
        print(mu, s, "rec-min", mr, "norm-min", mq, "spur_rec", sp_r, "coll", coll, "/", pcoll, "dist %.2f" % dist, "massrms %.3f" % rows[-1]["mass_rms"], est)
L.dump_json(L.HERE / "r2_mu_offset.json", dict(D2=D2, rows=rows))
