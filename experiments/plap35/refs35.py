"""References for p = 3.5, unit disk: radial shooting (E1, radial 2nd) and P1-FEM on nodal domains (disk [validation],
half disk = one nodal diameter, quarter disk).  -> experiments/results/plap35_refs.json   (resumable per domain)"""
import json, os
import torch
from experiments.plap.refs import radial_eigs
from experiments.plap import fem_plap as F
P = os.path.join(os.path.dirname(__file__), "..", "results", "plap35_refs.json")
torch.set_num_threads(1)
out = json.load(open(P)) if os.path.exists(P) else {}
v = [radial_eigs(3.5, 2, r0=r0, tol=t) for r0, t in [(1e-4, 1e-10), (1e-6, 1e-12), (1e-8, 1e-13)]]
out["shoot"] = dict(E1=v[1][0], E_rad2=v[1][1], spread=max(abs(a[0] - v[1][0]) for a in v))
print(out["shoot"], flush=True)
for dom in ["halfdisk", "disk", "sector_pi/2"]:
    if dom not in out:
        out[dom] = F.run(dom, 3.5, [20, 40, 80, 160])
        json.dump(out, open(P, "w"), indent=1)
