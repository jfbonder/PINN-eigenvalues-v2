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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    for job in jobs.QUEUES[a.worker]:
        if a.only and a.only != job["name"]:
            continue
        try:
            if "local_from" in job:
                job = resolve_local(job)
            core.run_job(job)
        except Exception as e:                                   # keep the queue alive
            import traceback; traceback.print_exc()
            print(f"[{job['name']}] FAILED: {e}", flush=True)
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
