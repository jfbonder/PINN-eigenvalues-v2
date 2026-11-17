"""Item 1 analysis: mu sensitivity table + figure (loss curves vs closed form)."""
import glob, json, math, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.mu_cost import lib as L

def dd(E): return L.dist_to_spec(E)
def closed(E, mu):
    d2 = dd(E) ** 2
    return d2 - d2 ** 2 / (4 * mu) if d2 <= 2 * mu else mu
def load(tag):
    d = L.load_json(L.RES / f"{tag}.json", None); return d

Es = np.arange(40, 60.1, 2.0)
D2 = max(dd(E) for E in np.arange(40, 60.001, 0.01)) ** 2
mus = [1, 10, 30, 60, 100, 113, 300, 1000, 10000]
rows = []
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True)
Ef = np.linspace(40, 60, 400)
for ax, mu in zip(axs.flat, mus):
    ax.plot(Ef, [closed(E, mu) for E in Ef], "k-", lw=1.2, label=r"closed form $d^2-d^4/4\mu$" )
    for s, c in ((0, "C0"), (1, "C3")):
        dat = load(f"mucost_mu{mu}_s{s}")
        if not dat or not dat["records"]: continue
        R = dat["records"]
        E = np.array([r["E"] for r in R]); tot = np.array([r["eval_loss_res"] + r["mu"] * r["eval_norm_err"] for r in R])
        ax.plot(E, tot, "o", color=c, ms=4, label=f"PINN total loss, seed {s}")
    ax.set_title(rf"$\mu={mu}$"); ax.set_yscale("symlog", linthresh=1)
    if mu == mus[0]: ax.legend(fontsize=6)
for ax in axs[-1]: ax.set_xlabel("E")
fig.suptitle(r"Total penalized loss at the trained minimizer vs. closed form (window [40,60], reduced net $3\times64$)")
fig.tight_layout(); fig.savefig(L.HERE / "figures/mu_loss_curves.png", dpi=130); plt.close(fig)

# per-mu summary
out = []
for mu in mus:
    for s in (0, 1):
        dat = load(f"mucost_mu{mu}_s{s}")
        if not dat or len(dat["records"]) < len(Es): continue
        R = dat["records"]
        E = np.array([r["E"] for r in R]); res = np.array([r["eval_loss_res"] for r in R])
        mass = np.array([r["mass"] for r in R]); tot = np.array([r["eval_loss_res"] + r["mu"] * r["eval_norm_err"] for r in R])
        quot = res / np.maximum(mass, 1e-12)   # normalised quotient Lambda(u,E)
        rho = np.array([r["rq_lap"] for r in R]); rg = np.array([r["rq_grad"] for r in R])
        cl = np.array([closed(e, mu) for e in E])
        collapsed = mass < 0.1
        pm = np.array([max(0.0, 1 - dd(e) ** 2 / (2 * mu)) for e in E]); pred_collapsed = int((pm < 0.1).sum()); mass_rms = float(np.sqrt(np.mean((mass - pm) ** 2)))
        def locmin(y):
            return [i for i in range(1, len(y) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]
        mins_rec = locmin(res)   # recorded curve (what sweep() thresholds)
        mins_q = locmin(quot)
        # eigenvalue estimate: Rayleigh quotient at the node of the smallest normalised quotient (non-collapsed)
        ok = np.where(~collapsed)[0]
        if len(ok):
            i = ok[np.argmin(quot[ok])]
            Ehat = rho[i]; err = abs(Ehat - 5 * L.PI2) / (5 * L.PI2)
        else:
            Ehat = float("nan"); err = float("nan")
        # relative curve misfit to closed form on non-collapsed nodes
        mis = float(np.max(np.abs(tot - cl) / np.maximum(cl, 1.0)))
        out.append(dict(mu=mu, seed=s, mu_over_D2=mu / D2, Ehat=Ehat, relerr=err, n_collapsed=int(collapsed.sum()),
                        min_mass=float(mass.min()), mass_at_49=float(mass[np.argmin(abs(E - 50))]),
                        minima_recorded=[float(E[i]) for i in mins_rec], minima_quotient=[float(E[i]) for i in mins_q],
                        misfit=mis, pred_collapsed=pred_collapsed, mass_rms=mass_rms, res_misfit=float(np.max(np.abs(res - np.array([dd(e)**2*max(0.0,1-dd(e)**2/(2*mu)) for e in E])))), time_h=sum(r["wall_time"] for r in R) / 3600))
L.dump_json(L.HERE / "mu_summary.json", dict(D2=D2, rows=out))
print("D^2 =", D2)
for o in out: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})

fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True)
for ax, mu in zip(axs.flat, mus):
    ax.plot(Ef, [max(0.0, 1 - dd(E) ** 2 / (2 * mu)) for E in Ef], "k-", lw=1.2, label=r"predicted $1-d^2/2\mu$ (floor 0)")
    for s_, c in ((0, "C0"), (1, "C3")):
        dat = load(f"mucost_mu{mu}_s{s_}")
        if dat and dat["records"]:
            ax.plot([r["E"] for r in dat["records"]], [r["mass"] for r in dat["records"]], "o", color=c, ms=4, label=f"mean u^2, seed {s_}")
    ax.set_title(rf"$\mu={mu}$")
    if mu == mus[0]: ax.legend(fontsize=6)
for ax in axs[-1]: ax.set_xlabel("E")
fig.suptitle("Norm of the trained network vs. predicted norm (collapse u=0 when d^2 >= 2 mu)")
fig.tight_layout(); fig.savefig(L.HERE / "figures/mu_mass.png", dpi=130)
