"""Collect all results -> report_data.json, figures, tables.tex (booktabs)."""
import json, math, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.mu_cost import lib as L
from experiments.mu_cost.analyze_scan import analyse_chain, nearest, gap_of
torch.set_num_threads(1)
EX = L.exact_eigs(60)
FIG = L.HERE / "figures"
out = {}

# ---------- item 4 (+3): paper-protocol sweep and cold chains ----------
sw = analyse_chain("mucost_t1_sweep")
out["sweep"] = sw
rows = []
if sw:
    R = L.load_json(L.RES / "mucost_t1_sweep.json", {})["records"]
    E = np.array([r["E"] for r in R]); res = np.array([r["eval_loss_res"] for r in R]); r2 = np.array([r["r"] ** 2 for r in R])
    mass = np.array([r["mass"] for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].semilogy(E, res, "o-", ms=3, label=r"recorded $\bar\Lambda$ residual term (fresh sample)")
    ax[0].semilogy(E, r2, "s-", ms=3, label=r"normalised quotient $r^2$")
    for e in sorted(set(np.round(EX[:12], 6))): ax[0].axvline(e, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("E"); ax[0].legend(fontsize=7); ax[0].set_title("paper-protocol scan, 80 nodes on [0,105]")
    ax[1].plot(E, mass, "o-", ms=3); ax[1].set_xlabel("E"); ax[1].set_title(r"mean $u^2$ (norm) at the trained network")
    fig.tight_layout(); fig.savefig(FIG / "paper_sweep_curve.png", dpi=130); plt.close(fig)

# ---------- all refinement comparisons ----------
chains = ["mucost_t1_sweep", "mucost_t1_11", "mucost_t1_12", "mucost_scan_broad_mu450_s0", "mucost_scan_broad_mu450_s1",
          "mucost_scan_broad_mu450_s2", "mucost_t1_narrow"]
ref = []
for t in chains:
    a = out["sweep"] if t == "mucost_t1_sweep" else analyse_chain(t)
    if not a: continue
    out[t] = a
    for m in a["minima"]:
        if m.get("err_old") is not None: ref.append(dict(chain=t, **m))
out["refine_rows"] = ref
L.dump_json(L.HERE / "report_data.json", out)
if ref:
    keys = [("old", "re-sampled Brent (refine())"), ("rq_lap_mc", "closed-form RQ, fixed MC 1e5"),
            ("rq_lap_grid", "closed-form RQ, midpoint grid 300^2"), ("rq_grad_grid", "gradient-form RQ, grid"),
            ("parab", "3-pt parabola of r^2"), ("parab_res", "3-pt parabola of recorded residual")]
    print("n minima compared:", len(ref))
    for k, name in keys:
        v = np.array([m["err_" + k] for m in ref if m.get("err_" + k) is not None])
        print(f"{name:42s} median {np.median(v):.2e}  max {v.max():.2e}  mean {v.mean():.2e}")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, (k, name) in enumerate(keys):
        v = [max(m["err_" + k], 1e-7) for m in ref if m.get("err_" + k) is not None]
        ax.semilogy(np.full(len(v), j) + np.linspace(-.15, .15, len(v)), v, "o", ms=4)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k for k, _ in keys], rotation=20, fontsize=7)
    ax.set_ylabel("|E_hat - E_exact|"); ax.set_title(f"refinement estimators on {len(ref)} detected minima")
    fig.tight_layout(); fig.savefig(FIG / "refine_compare.png", dpi=130); plt.close(fig)

# ---------- item 2: seeds overlay ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for s, c in ((0, "C0"), (1, "C1"), (2, "C2")):
    d = L.load_json(L.RES / f"mucost_scan_broad_mu450_s{s}.json", None)
    if not d: continue
    R = [r for r in d["records"] if r["E"] <= 55]
    E = [r["E"] for r in R]
    ax[0].semilogy(E, [r["r"] ** 2 for r in R], "o-", color=c, ms=4, label=f"seed {s}")
    ax[1].plot(E, [r["mass"] for r in R], "o-", color=c, ms=4, label=f"seed {s}")
Ef = np.linspace(15, 55, 400)
ax[0].semilogy(Ef, [L.dist_to_spec(e) ** 2 for e in Ef], "k-", lw=1, label=r"$d(E)^2$ (exact)")
ax[0].set_xlabel("E"); ax[0].set_title(r"normalised quotient $r^2$ (fresh sample), window [15,55], $\mu=450$"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("E"); ax[1].set_title("mean $u^2$"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "seeds_overlay.png", dpi=130); plt.close(fig)
print("report data written")
