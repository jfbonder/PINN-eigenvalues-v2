"""Figures + tables.tex + README tables from results JSONs."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.fem import pinn_analysis as PA

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
S = json.load(open(RES / "fem_summary.json"))

def fig_spectra():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = [("ball_free", "free ball (exact)"), ("double_well", "double well"), ("four_well", "four well")]
    for y, (c, lab) in enumerate(reversed(names)):
        for r in S[c]["rows"]:
            ax.plot([r["E"]] * 2, [y - 0.35, y + 0.35], color=f"C{2-y}", lw=1.2 + 0.7 * (r["mult"] - 1) ** 0.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([n[1] for n in reversed(names)])
    ax.set_xlabel("E"); ax.set_xlim(0, 75); ax.set_title("FEM spectra in the unit ball, E in [0,75] (line width ~ multiplicity)")
    fig.tight_layout(); fig.savefig(FIG / "fem_spectra_ball.png", dpi=150); plt.close(fig)

def fig_convergence():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axs[0]
    for c, lab in [("disk_free", "disk, curved P2"), ("annulus", "annulus, curved P2"), ("square", "square, P2")]:
        s = S[c]; n = np.array(s["ndof"], float)
        err = [abs(np.load_vals(c)) if False else None for _ in n]
    lv = {}
    def levels(c):
        d = json.load(open(RES / f"fem_{c}.json"))["levels"]; ks = sorted(d, key=int)
        return [d[k]["ndof"] for k in ks], [np.array(d[k]["eigenvalues"]) for k in ks]
    for c, lab, kk in [("disk_free", "disk (E=5.78)", 0), ("annulus", "annulus (E=39.01)", 0), ("square", "square (E=19.74)", 0),
                       ("harm_w10", r"harmonic $\omega=10$ (E=14.28)", 0)]:
        nd, ev = levels(c); ref = S[c]["rows"][0].get("exact", S[c]["rows"][0]["E"])
        ax.loglog(nd, [abs(e[kk] - ref) + 1e-14 for e in ev], "o-", label=lab)
    ax.set_xlabel("dof"); ax.set_ylabel("|E_h - E_ref|"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("2-D, P2 (order 4 in h)", fontsize=9)
    ax = axs[1]
    nd, ev = levels("ball_free")
    ex = [r.get("exact") for r in S["ball_free"]["rows"]]
    for k, lab in [(0, "E1=9.87"), (1, "E2=20.19"), (3, "E=39.48")]:
        idx = {0: 0, 1: 1, 3: 9}[k]
        e_ex = ex[k]
        ax.loglog(nd, [abs(e[idx] - e_ex) for e in ev], "o-", label="ball " + lab)
    nd, ev = levels("double_well")
    ref = S["double_well"]["rows"][0]["E"]
    ax.loglog(nd[:-1], [abs(e[0] - ref) for e in ev[:-1]], "s--", label="double well E1 (vs extrap.)")
    ax.set_xlabel("dof"); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title("3-D ball, curved P2 tets (order ~2.8 in h)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fem_convergence.png", dpi=150); plt.close(fig)

def fig_windows(case, title):
    summ, res = PA.load_all(case)
    if not res: return None
    s0 = sorted(res)[0]
    wins = {k: w for k, w in res[s0]["windows"].items() if w}
    if not wins: return None
    n = len(wins); fig, axs = plt.subplots(1, n, figsize=(3.3 * n, 3.1), squeeze=False)
    for ax, (k, w) in zip(axs[0], sorted(wins.items(), key=lambda t: float(t[0]))):
        E = np.array(w["E"])
        ax.plot(E, w["loss_res"], "o-", ms=3, label="recorded loss (residual term)")
        ax.plot(E, w["r2"], "s-", ms=3, label=r"normalised $\Lambda=\overline{res^2}/\overline{u^2}$")
        xs = np.linspace(E.min(), E.max(), 300)
        fe = [r["E"] for r in summ if E.min() - 2 < r["E"] < E.max() + 2]
        if fe:
            d2 = np.min((xs[:, None] - np.array(fe)[None]) ** 2, axis=1)
            ax.plot(xs, d2, "k:", lw=1, label=r"$\min_k(E-E_k^{FEM})^2$")
        for r in summ:
            if E.min() - 0.3 <= r["E"] <= E.max() + 0.3:
                ax.axvline(r["E"], color="r", lw=0.8, ls="--")
                ax.text(r["E"], ax.get_ylim()[1] * 0.97, f"x{r['mult']}" if r["mult"] > 1 else "", color="r", fontsize=7, ha="center", va="top")
        ax.set_xlabel("E"); ax.set_title(f"window around {float(k):.2f}", fontsize=9); ax.grid(alpha=.3)
    axs[0][0].set_ylabel("loss"); axs[0][0].legend(fontsize=6)
    fig.suptitle(title + " (red dashed: FEM eigenvalues)", fontsize=10); fig.tight_layout()
    fig.savefig(FIG / f"pinn_vs_fem_{case}.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    fig_spectra(); fig_convergence()
    for c, t in [("double_well", "Double well, 3-D ball"), ("four_well", "Four-well bump, 3-D ball"), ("harm_w10", r"Harmonic disk, $\omega=10$")]:
        fig_windows(c, t)
    print("figures done")
