"""Items 4, 3, 2, 5(our side): analyse scan chains. Writes scan_summary.json and prints tables."""
import sys, math
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.mu_cost import lib as L
torch.set_num_threads(1)
EX = L.exact_eigs(60)

def nearest(E):
    e = min(EX, key=lambda x: abs(x - E)); return e
def gap_of(ek):
    return min(abs(ek - x) for x in set(np.round(EX, 9)) if abs(x - ek) > 1e-6)

def node_model(tag, i, hidden=128):
    m = L.make_model(hidden); m.load_state_dict(torch.load(L.CK / f"{tag}_node{i}.pt")); m.eval(); return m

def analyse_chain(tag, hidden=128, with_cert=True):
    d = L.load_json(L.RES / f"{tag}.json", None)
    if not d: return None
    R = d["records"]; E = np.array([r["E"] for r in R])
    r2 = np.array([r["r"] ** 2 for r in R]); res = np.array([r["eval_loss_res"] for r in R]); mass = np.array([r["mass"] for r in R])
    # minima of the NORMALISED curve (r^2) and of the recorded curve (res)
    def locmin(y): return [i for i in range(1, len(y) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]
    mins_r = locmin(r2); mins_rec = locmin(res)
    out = dict(tag=tag, config=d["config"], wall_total=sum(r["wall_time"] for r in R),
               refine_total=sum(r.get("refine_time", 0) for r in R), n_nodes=len(R),
               epochs=[r["epochs_run"] for r in R], minima_norm=[float(E[i]) for i in mins_r],
               minima_rec=[float(E[i]) for i in mins_rec], minima=[])
    for i in mins_r:
        h = E[i + 1] - E[i]
        ek = nearest(E[i])
        est = dict(node=int(i), E_node=float(E[i]), exact=ek,
                   old=R[i].get("ref_old_E"), rq_lap_grid=R[i].get("rq_lap_grid"), rq_grad_grid=R[i].get("rq_grad_grid"),
                   rq_lap_mc=R[i]["rq_lap"], parab=L.parab_vertex(E[i - 1:i + 2], r2[i - 1:i + 2]),
                   parab_res=L.parab_vertex(E[i - 1:i + 2], res[i - 1:i + 2]), node_r=R[i]["r"], node_res=R[i]["eval_loss_res"],
                   mass=R[i]["mass"])
        for k in ("old", "rq_lap_grid", "rq_grad_grid", "rq_lap_mc", "parab", "parab_res"):
            v = est[k]; est["err_" + k] = None if v is None else abs(v - ek)
        if with_cert and (L.CK / f"{tag}_node{i}.pt").exists():
            m = node_model(tag, i, hidden)
            Eref = est["rq_lap_grid"]
            torch.manual_seed(999)
            ev = L.fresh_eval(m, Eref, n=50000)          # certificate at the reported value (fresh sample)
            est.update(E_report=Eref, cert_res=ev["loss_res"], cert_r=ev["r"], cert_err=abs(Eref - ek),
                       weinstein_ok=bool(abs(Eref - ek) <= ev["r"]), gap=gap_of(ek),
                       kato_temple=ev["r"] ** 2 / gap_of(ek), rel_err=abs(Eref - ek) / ek)
            # certificate for the old refine() value at ITS own E
            if est["old"] is not None:
                ev2 = L.fresh_eval(m, est["old"], n=50000)
                est.update(old_cert_r=ev2["r"], old_weinstein_ok=bool(abs(est["old"] - ek) <= ev2["r"]), old_cert_res=ev2["loss_res"])
        out["minima"].append(est)
    return out

if __name__ == "__main__":
    tags = sys.argv[1:]
    allr = {}
    for t in tags:
        a = analyse_chain(t)
        if a is None: continue
        allr[t] = a
        print("==", t, "wall %.0fs (refine %.0fs)" % (a["wall_total"], a["refine_total"]), "minima(norm):", a["minima_norm"], "minima(rec):", a["minima_rec"])
        for m in a["minima"]:
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
    L.dump_json(L.HERE / "scan_summary.json", allr)
