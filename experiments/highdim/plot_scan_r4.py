"""Plot the full warm-started scan in the unit ball of R^4 (paper Fig. loss3D, right panel).

usage: python -m experiments.highdim.plot_scan_r4 [--seed 0]
reads  experiments/results/scan_r4_s<seed>.json   (written by experiments/highdim/scan_r4.py)
writes paper/v4/loss_ball4_scan_mu200.png and prints the local minima of the recorded residual.

Curves: recorded residual term R_mu(E) (loss_res returned by train_one, i.e. what the scan thresholds),
normalized loss r^2 (Monte Carlo evaluation on fresh points, mc_eval), and the exact curve l(E)^2,
l(E) = dist(E, sigma), with the Dirichlet eigenvalues j_{m+1,n}^2 of the unit 4-ball.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jn_zeros

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "paper" / "v4" / "loss_ball4_scan_mu200.png")); a = ap.parse_args()

d = json.load(open(ROOT / "experiments" / "results" / f"scan_r4_s{a.seed}.json"))
R = sorted(d["records"], key=lambda r: r["E"])
E = np.array([r["E"] for r in R]); res = np.array([r["loss_res"] for r in R])
r2 = np.array([r["r"] ** 2 for r in R]); rho = np.array([r["rho_res"] for r in R]); mass = np.array([r["mass"] for r in R])

# Dirichlet eigenvalues of the unit ball in R^4: j_{m+1,n}^2 (multiplicity (m+1)^2)
ev = np.unique(np.round(np.concatenate([jn_zeros(m + 1, 6) ** 2 for m in range(12)]), 6))
ev = ev[ev < 120]
Ef = np.linspace(0, 56, 2000)
ell2 = np.min((Ef[:, None] - ev[None, :]) ** 2, axis=1)

mins = [i for i in range(1, len(E) - 1) if res[i] < res[i - 1] and res[i] < res[i + 1]]
print("eigenvalues of the 4-ball below 60:", ", ".join(f"{x:.4f}" for x in ev[ev < 60]))
print("local minima of the recorded residual (node E, R_mu, r^2, Rayleigh quotient, mass, nearest eigenvalue):")
for i in mins:
    nn = ev[np.argmin(abs(ev - rho[i]))]
    print(f"  E={E[i]:6.2f}  R_mu={res[i]:8.4f}  r^2={r2[i]:8.4f}  rho={rho[i]:9.4f}  mass={mass[i]:.3f}  "
          f"nearest={nn:.4f}  rel.diff={100 * (rho[i] - nn) / nn:+.3f}%")
mins_r2 = [i for i in range(1, len(E) - 1) if r2[i] < r2[i - 1] and r2[i] < r2[i + 1]]
print("local minima of r^2 at E =", [float(E[i]) for i in mins_r2])

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(Ef, ell2, "-", color="tab:orange", lw=1.5, label=r"$\ell(E)^2$ (exact)")
ax.plot(E, r2, "s--", color="tab:green", ms=3, lw=0.8, label=r"normalized loss $r^2$")
ax.plot(E, res, "o-", color="tab:blue", ms=3, lw=1.2, label=r"recorded residual $\mathcal{R}_\mu(E)$")
ax.plot(E[mins], res[mins], "v", color="tab:red", ms=8, label="local minima")
for x in ev[ev < 56]:
    ax.axvline(x, color="gray", ls=":", lw=0.8)
ax.set_xlim(0, 56); ax.set_ylim(0, 210)
ax.set_xlabel(r"$E$"); ax.set_ylabel("Loss")
ax.text(0.97, 0.62, r"$d=4$", transform=ax.transAxes, ha="right", fontsize=12)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(a.out, dpi=130)
print("wrote", a.out)
