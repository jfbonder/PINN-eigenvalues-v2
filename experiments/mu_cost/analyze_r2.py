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

# ---------- (c) cold start at 10 pi^2 ----------
print("\n== cold start [95,102] ==")
cold = {}
for mu in (3000, 450, 5000, 20000):
    for s in (0, 1, 2):
        tag = {3000: "mucost_t1_13", 450: "mucost_t1_narrow"}.get(mu, f"mucost_t1_cold_mu{mu}_s{s}")
        if mu in (3000, 450) and s > 0: continue
        if mu == 5000 and s == 0: tag = "mucost_t1_narrow5000"
        d = L.load_json(L.RES / f"{tag}.json", None)
        if not d or len(d["records"]) < 5: continue
        R = d["records"]
        rho = [r["rq_lap"] for r in R]; ms = [r["mass"] for r in R]
        conv = [near(x) for x in rho]
        hit = [bool(abs(x - 10 * L.PI2) < 0.05 and m > 0.5) for x, m in zip(rho, ms)]
        branch = [round(float(c / L.PI2), 2) for c in conv]
        cold[f"{mu}_s{s}"] = dict(tag=tag, rq=rho, mass=ms, hit=hit, wall=sum(r["wall_time"] for r in R), epochs=[r["epochs_run"] for r in R])
        print(mu, s, "RQ per node", [round(x, 2) for x in rho], "mass", [round(m, 2) for m in ms], "hit(10pi^2)", hit, "wall %.0f" % cold[f"{mu}_s{s}"]["wall"])
L.dump_json(L.HERE / "r2_cold.json", cold)

# ---------- (c) delta = |r^2 - d^2| ----------
print("\n== delta ==")
dl = {}
def delta(tag, Emax=1e9):
    d = L.load_json(L.RES / f"{tag}.json", None)
    if not d: return
    R = [r for r in d["records"] if r["E"] <= Emax]
    rows = []
    for r in R:
        coll = r["mass"] < 0.1; sw = (near(r["rq_lap"]) != near(r["E"])) and not coll
        rows.append((r["E"], abs(r["r"] ** 2 - dd(r["E"]) ** 2), coll, sw))
    ok = [x for x in rows if not x[2] and not x[3]]
    out = dict(n=len(R), max_all=max(x[1] for x in rows), n_excl=len(R) - len(ok), n_collapsed=sum(x[2] for x in rows),
               n_switch=sum(x[3] for x in rows), max_clean=max(x[1] for x in ok), med_clean=float(np.median([x[1] for x in ok])))
    dl[tag + (f"<= {Emax}" if Emax < 1e8 else "")] = out; print(tag, Emax, out)
delta("mucost_t1_sweep")
for s in (0, 1, 2): delta(f"mucost_scan_broad_mu450_s{s}", 55)
for s in (0, 1, 2): delta(f"mucost_scan_broad_off_s{s}")
L.dump_json(L.HERE / "r2_delta.json", dl)

# ---------- (b) cost ----------
print("\n== cost ==")
cost = {}
for s in (0, 1, 2):
    a = L.load_json(L.RES / f"mucost_scan_broad_off_s{s}.json", None)
    if not a or len(a["records"]) < 18: continue
    R = a["records"]; E = np.array([r["E"] for r in R]); r2 = np.array([r["r"] ** 2 for r in R]); mass = np.array([r["mass"] for r in R])
    mins = [i for i in range(1, len(R) - 1) if r2[i] < r2[i - 1] and r2[i] < r2[i + 1] and mass[i] > 0.5]
    est = []
    for i in mins:
        ek = near(E[i]); est.append((float(E[i]), ek, float(R[i].get("rq_lap_grid", R[i]["rq_lap"])), abs(R[i].get("rq_lap_grid", R[i]["rq_lap"]) - ek) / ek))
    cost[f"scan_s{s}"] = dict(wall=sum(r["wall_time"] for r in R), refine=sum(r.get("refine_time", 0) for r in R), n=len(R), minima=est,
                               epochs=sum(r["epochs_run"] for r in R))
    print("scan", s, cost[f"scan_s{s}"]["wall"], est)
for s in (0, 1, 2):
    for v in ("es", "noes"):
        a = L.load_json(L.RES / f"mucost_r2_rq_s{s}_{v}.json", None)
        if not a or len(a["records"]) < 6: continue
        R = a["records"]
        cost[f"rq_s{s}_{v}"] = dict(wall=sum(r["wall_time"] for r in R), wall5=sum(r["wall_time"] for r in R[:5]),
                                    err=[r["rel_err_vs_nearest"] for r in R], epochs=[r["epochs_run"] for r in R])
        print("rq", s, v, cost[f"rq_s{s}_{v}"]["wall"], ["%.1e" % x for x in cost[f"rq_s{s}_{v}"]["err"]])
L.dump_json(L.HERE / "r2_cost.json", cost)
