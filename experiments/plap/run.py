"""Queue worker:  python -m experiments.plap.run --worker {0,1} [--only NAME]   (1 torch thread, resumable)."""
import argparse, json
import numpy as np
import torch
from experiments.plap import core, jobs


def minima(E, L):
    """Indices of interior local minima of the curve, in increasing E."""
    return [i for i in range(1, len(L) - 1) if L[i] < L[i - 1] and L[i] < L[i + 1]]


def resolve_local(job):
    src = json.load(open(core.RES / f"plap_{job['local_from']}.json"))["records"]
    E = np.array([r["E"] for r in src]); L = np.array([r["Lam_n"] for r in src])
    mins = [i for i in minima(E, L) if L[i] < 0.5 * np.median(L)]
    i = mins[job["k"]]
    c = np.polyfit(E[i - 1:i + 2], L[i - 1:i + 2], 2); Ev = -c[1] / (2 * c[0])
    h = job["h"]; Ec = round(Ev / (h / 2)) * (h / 2)             # centre snapped to a multiple of h/2
    m = job["npts"] // 2
    job = dict(job); job["E"] = [float(Ec + h * j) for j in range(-m, m + 1)]
    job["coarse_vertex"] = float(Ev)
    return job


def merged_curve(names):
    recs = {}
    for n in names:
        for r in json.load(open(core.RES / f"plap_{n}.json"))["records"]:
            recs.setdefault(r["E"], (r, n))                       # first file wins on overlaps
    E = np.array(sorted(recs)); L = np.array([recs[e][0]["Lam_n"] for e in E])
    return E, L, [recs[e][1] for e in E]


def detect(E, L):
    """Interior local minima that are at least 30% below the largest value among the 2 neighbours on each side."""
    return [i for i in minima(E, L) if L[i] < 0.7 * max(L[max(i - 2, 0):i + 3])]


def run_local_all(job):
    E, L, src = merged_curve(job["local_all_from"])
    for k, i in enumerate(detect(E, L)):
        c = np.polyfit(E[i - 1:i + 2], L[i - 1:i + 2], 2); Ev = -c[1] / (2 * c[0])
        h = job["h"]; Ec = round(Ev / (h / 2)) * (h / 2); m = job["npts"] // 2
        sub = {a: b for a, b in job.items() if a != "local_all_from"}
        sub.update(name=f"{job['name']}{k}", E=[float(Ec + h * j) for j in range(-m, m + 1)], coarse_vertex=float(Ev),
                   coarse_E=float(E[i]), init_ckpt=str(core.CKPT / f"{src[i]}_E{E[i]:.4f}.pt"))
        import os
        if not os.path.exists(sub["init_ckpt"]):
            sub["epochs0"] = 3000                                 # no coarse network stored: start from scratch, longer
        core.run_job(sub)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    for job in jobs.QUEUES[a.worker]:
        if a.only and a.only != job["name"]:
            continue
        try:
            if "local_all_from" in job:
                run_local_all(job); continue
            if "local_from" in job:
                job = resolve_local(job)
            core.run_job(job)
        except Exception as e:                                   # keep the queue alive
            import traceback; traceback.print_exc()
            print(f"[{job['name']}] FAILED: {e}", flush=True)
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
