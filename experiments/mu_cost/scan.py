"""Resumable warm-start scan on the unit square.
python -m experiments.mu_cost.scan --tag T --E 40:60:2 --mu 1000 --seed 0 [--hidden 128 --epochs 3000 --npts 5000]
--E accepts start:stop:step or a comma list."""
import argparse, numpy as np, torch
from experiments.mu_cost import lib as L
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--E", required=True)
ap.add_argument("--mu", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--epochs", type=int, default=3000)
ap.add_argument("--npts", type=int, default=5000); ap.add_argument("--patience", type=int, default=300)
a = ap.parse_args()
torch.set_num_threads(1)
if ":" in a.E:
    s, e, h = map(float, a.E.split(":")); Es = np.arange(s, e + 1e-9, h)
else:
    Es = np.array([float(x) for x in a.E.split(",")])
L.run_chain(a.tag, Es, a.mu, a.seed, hidden=a.hidden, n_epochs=a.epochs, n_samples=a.npts, patience=a.patience)
