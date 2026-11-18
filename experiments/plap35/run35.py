"""python -m experiments.plap35.run35 --queue A|B [--names a,b]   (1 thread, resumable)"""
import argparse, traceback
import torch
from experiments.plap35 import core35, jobs35
ap = argparse.ArgumentParser(); ap.add_argument("--queue", default="A"); ap.add_argument("--names", default="")
a = ap.parse_args(); torch.set_num_threads(1)
q = jobs35.QUEUES[a.queue]
if a.names:
    allj = {j["name"]: j for v in jobs35.QUEUES.values() for j in v}; q = [allj[n] for n in a.names.split(",")]
for job in q:
    try:
        if job["E"] == "E2":
            job = dict(job, E=jobs35.e2_grid())
        core35.run_job(job)
    except Exception as e:
        traceback.print_exc(); print(f"[{job['name']}] FAILED {e}", flush=True)
print("QUEUE DONE", flush=True)
