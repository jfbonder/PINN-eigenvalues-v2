"""Tables + figures for the p = 3.5 study.  python -m experiments.plap35.analyze35  -> tables35.md, figures/, plap35_summary.json"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.plap35 import core35 as C, jobs35

HERE = Path(__file__).resolve().parent; RES = C.RES; FIG = HERE / "figures"
R = json.load(open(RES / "plap35_refs.json")); E1 = R["shoot"]["E1"]; E2 = R["halfdisk"]["extrap"]
pc = lambda a, ref=E1: f"{100*(a-ref)/ref:+.2f}"


def load(n):
    f = RES / f"plap35_{n}.json"
    return json.load(open(f)) if f.exists() else None


def vertex(E, L, i):
    i = min(max(i, 1), len(E) - 2); c = np.polyfit(E[i-1:i+2], L[i-1:i+2], 2); return -c[1] / (2 * c[0])


def fixed_point(E, Ev, ok=None):
    """Self-consistent value: E* with E_PINN(E*) = E*, by linear interpolation between the two nodes where E_PINN(E) - E changes sign."""
    E = np.asarray(E, float); f = np.asarray(Ev, float) - E
    for i in range(len(E) - 1):
        if ok is not None and not (ok[i] and ok[i + 1]): continue
        if f[i] > 0 >= f[i + 1]:
            return float(E[i] + f[i] * (E[i + 1] - E[i]) / (f[i] - f[i + 1]))
    return float("nan")


DESC = {"base_s0": "baseline (strong form), P1, seed 0", "base_s1": "baseline, seed 1", "base_s2": "baseline, seed 2",
        "base_P0_s0": "baseline, P0 (const lr + early stop), seed 0", "base_P0_s1": "baseline, P0, seed 1", "base_P0_s2": "baseline, P0, seed 2",
        "base_P0c_s0": "baseline, constant lr, 1500 epochs", "base_N20000": "(c) N = 20000", "base_imp": "(c) importance sampling (50 % in r<0.3)",
        "base_mu1000": "(d) mu = 1000", "base_ep5000": "(d) 5000 epochs",
        "mixed_s0": "(a) mixed, lam=100, seed 0", "mixed_s1": "(a) mixed, lam=100, seed 1", "mixed_s2": "(a) mixed, lam=100, seed 2",
        "mixed_lam10": "(a) mixed, lam=10, seed 0", "mixed_lam10_s1": "(a) mixed, lam=10, seed 1", "mixed_lam10_s2": "(a) mixed, lam=10, seed 2",
        "mixed_lam1000": "(a) mixed, lam=1000", "mixed_P0_s0": "(a) mixed, lam=100, P0",
        "wdown_k0.3": "(b) down-weighting omega=|Du|^2/(|Du|^2+0.09)", "wdiv_k0.3": "(b') divide by (|Du|^2+0.09)^{(p-2)/2}",
        "feat0_s0": "(e) singular feature |x|^{p/(p-1)}, seed 0", "feat0_s1": "(e) singular feature, seed 1", "feat0_s2": "(e) singular feature, seed 2",
        "featc_s0": "(e) singular feature, trainable centre"}
S = dict(E1=E1, E2_halfdisk=E2, rows=[]); md = []
md.append(f"References: E1 = {E1:.5f} (radial shooting; P1-FEM {R['disk']['extrap']:.5f}); half disk (1 nodal diameter) {E2:.4f}; "
          f"quarter disk {R['sector_pi/2']['extrap']:.3f}; radial 2nd {R['shoot']['E_rad2']:.2f}.\n")
md.append("### Table 1: 5-node window (E = 10.5 … 14.5, h = 1; N20000/ep5000: 3 nodes 11.5–13.5).  loss = variant functional on a fresh sample of 40000\n")
md.append("| variant | loss at nodes | grid min | 3-pt vertex (err %) | E_PINN at node 12.5 (err %) | E_PINN over nodes 11.5–13.5 (err %) | core frac @12.5 | MC std of loss @12.5 (batches of 5000) | collapsed nodes | wall min | self-consistent E* (err %) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for n, dsc in DESC.items():
    d = load(n)
    if d is None or len(d["records"]) < 3: continue
    r = d["records"]; E = np.array([x["E"] for x in r]); L = np.array([x["loss_var"] for x in r])
    ok = np.array([not x["collapsed"] for x in r])
    j = int(np.argmin(np.abs(E - 12.5))); i = int(np.argmin(np.where(ok, L, np.inf)))
    v = vertex(E, L, i)
    cen = [x["E_var"] for x in r if 11.4 < x["E"] < 13.6 and not x["collapsed"]]
    row = dict(name=n, desc=dsc, E=E.tolist(), loss=L.tolist(), grid_min=float(E[i]), vertex=float(v), E_var_125=r[j]["E_var"],
               E_var_nodes=[x["E_var"] for x in r], E_strong_nodes=[x["E_strong"] for x in r], core=r[j]["core_frac"], mc_std=r[j]["mc_std_var"],
               Lam_strong=[x["Lam_strong"] for x in r], collapsed=[x["E"] for x in r if x["collapsed"]], wall_min=d["total_wall_s"] / 60,
               E_weak=[x.get("E_weak") for x in r])
    row["fp"] = fixed_point(E, row["E_var_nodes"], ok)
    S["rows"].append(row)
    md.append(f"| {dsc} | " + ", ".join(f"{a:.2f}" if a < 1e3 else f"{a:.0e}" for a in L) + f" | {E[i]} | {v:.3f} ({pc(v)}) | {r[j]['E_var']:.3f} ({pc(r[j]['E_var'])}) | "
              f"{min(cen):.2f}–{max(cen):.2f} ({pc(min(cen))} to {pc(max(cen))}) | {r[j]['core_frac']:.2f} | {r[j]['mc_std_var']:.3f} | {row['collapsed'] or '–'} | {row['wall_min']:.0f} | {row['fp']:.3f} ({pc(row['fp'])}) |")

# seed statistics
def grp(names, key):
    v = [x[key] for x in S["rows"] if x["name"] in names]; return v
md.append("\n### Table 2: three-seed summary (node 12.5)\n\n| variant | loss(12.5) per seed | E_PINN(12.5) per seed | err % | 3-pt vertex per seed | err % | self-consistent E* per seed | err % |\n|---|---|---|---|---|---|---|---|")
for lab, names in [("baseline P1", ["base_s0", "base_s1", "base_s2"]), ("baseline P0", ["base_P0_s0", "base_P0_s1", "base_P0_s2"]),
                   ("(a) mixed lam=100", ["mixed_s0", "mixed_s1", "mixed_s2"]), ("(a) mixed lam=10", ["mixed_lam10", "mixed_lam10_s1", "mixed_lam10_s2"]),
                   ("(e) singular feature", ["feat0_s0", "feat0_s1", "feat0_s2"])]:
    rows = [x for x in S["rows"] if x["name"] in names]
    if not rows: continue
    md.append(f"| {lab} | " + ", ".join(f"{x['loss'][x['E'].index(12.5)]:.3f}" for x in rows) + " | " + ", ".join(f"{x['E_var_125']:.3f}" for x in rows) + " | "
              + ", ".join(pc(x['E_var_125']) for x in rows) + " | " + ", ".join(f"{x['vertex']:.3f}" for x in rows) + " | " + ", ".join(pc(x['vertex']) for x in rows) + " | "
              + ", ".join(f"{x['fp']:.3f}" for x in rows) + " | " + ", ".join(pc(x['fp']) for x in rows) + " |")

# fine curves
md.append("\n### Table 3: fine scans (h = 0.25, E = 11 … 14, warm start, seed 0)\n\n| variant | loss at the 13 nodes | argmin (err %) | nodes within 25 % of the minimum | E_PINN at argmin (err %) | MC std @argmin | self-consistent E* (err %) |\n|---|---|---|---|---|---|---|")
fine = {}
for n, lab in [("base_fine_P0", "baseline P0"), ("base_fine_P1", "baseline P1"), ("mixed_fine_P1", "(a) mixed lam=100 P1"), ("feat0_fine", "(e) singular feature P1")]:
    d = load(n)
    if d is None or len(d["records"]) < 5: continue
    r = d["records"]; E = np.array([x["E"] for x in r]); L = np.array([x["loss_var"] for x in r]); i = int(np.argmin(L))
    flat = E[L < 1.25 * L[i]]
    fine[n] = (E, L, np.array([x["mc_std_var"] for x in r]), lab, r)
    S["fine_" + n] = dict(E=E.tolist(), loss=L.tolist(), argmin=float(E[i]), flat=[float(flat.min()), float(flat.max())], E_var=[x["E_var"] for x in r])
    md.append(f"| {lab} | " + ", ".join(f"{a:.2f}" for a in L) + f" | {E[i]} ({pc(E[i])}) | {flat.min()}–{flat.max()} | {r[i]['E_var']:.3f} ({pc(r[i]['E_var'])}) | {r[i]['mc_std_var']:.3f} | {fixed_point(E, [x['E_var'] for x in r]):.3f} ({pc(fixed_point(E, [x['E_var'] for x in r]))}) |")

# E2
md.append(f"\n### Table 4: second eigenvalue window (reference {E2:.3f}, nodes from scratch)\n\n| run | nodes | loss | mass | E_PINN |\n|---|---|---|---|---|")
for n in ["base_E2", "mixed_E2", "base_E2_premass", "mixed_E2_premass"]:
    d = load(n)
    if d is None or not d["records"]: continue
    r = d["records"]
    md.append(f"| {n} | {[x['E'] for x in r]} | " + ", ".join(f"{x['loss_var']:.3g}" for x in r) + " | " + ", ".join(f"{x['mass']:.3f}" for x in r) + " | "
              + ", ".join(f"{x['E_var']:.2f}" for x in r) + " |")
    S["E2_" + n] = [dict(E=x["E"], loss=x["loss_var"], mass=x["mass"], E_var=x["E_var"], core=x["core_frac"]) for x in r]

# oscillation sources at node 12.5 (baseline)
md.append("\n### Table 5: sources of fluctuation of the loss at E = 12.5 (baseline strong form)\n\n| source | measure | value |\n|---|---|---|")
def tail_std(n, k=30):
    d = load(n)
    if d is None: return None
    r = [x for x in d["records"] if abs(x["E"] - 12.5) < 1e-9][0]; h = np.array(r["hist"])[-k:, 1]
    return h.mean(), h.std(), r
for n, lab in [("base_s0", "P1 (lr decay)"), ("base_P0c_s0", "constant lr"), ("base_P0_s0", "P0 (const lr + early stop)"), ("mixed_s0", "mixed P1"), ("feat0_s0", "singular feature P1")]:
    t = tail_std(n)
    if t is None: continue
    md.append(f"| training-batch residual, last {30 if t[2]['epochs_run']>300 else len(t[2]['hist'])} recorded epochs, {lab} | mean ± std (batch of 5000, moving network) | {t[0]:.2f} ± {t[1]:.2f} |")
    md.append(f"| same network frozen, {lab} | std over 8 fresh batches of 5000 / value on 40000 | {t[2]['mc_std_var']:.3f} / {t[2]['res_var']:.3f} (range {t[2]['mc_min_var']:.2f}–{t[2]['mc_max_var']:.2f}) |")
    S["fluct_" + n] = dict(train_tail_mean=float(t[0]), train_tail_std=float(t[1]), mc_std=t[2]["mc_std_var"], value=t[2]["res_var"])
for lab, names in [("seed-to-seed, P1", ["base_s0", "base_s1", "base_s2"]), ("seed-to-seed, P0", ["base_P0_s0", "base_P0_s1", "base_P0_s2"])]:
    v = [x["loss"][x["E"].index(12.5)] for x in S["rows"] if x["name"] in names]
    if v: md.append(f"| {lab} | loss(12.5) for 3 seeds | {', '.join(f'{a:.2f}' for a in v)} |")
for n, lab in [("base_s0", "1500 epochs at the node (h=1 scan)"), ("base_ep5000", "5000 epochs at the node"), ("base_fine_P1", "h=0.25 warm-started scan (6 previous nodes x 1500 epochs nearby)")]:
    d = load(n)
    if d:
        r = [x for x in d["records"] if abs(x["E"] - 12.5) < 1e-9]
        if r: md.append(f"| amount of training, {lab} | loss(12.5), core frac | {r[0]['loss_var']:.2f}, {r[0]['core_frac']:.2f} |")
json.dump(S, open(RES / "plap35_summary.json", "w"), indent=1)
open(HERE / "tables35.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

# ------------------------------- figures -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n, st, col in [("base_s0", "o-", "C0"), ("base_s1", "o-", "C0"), ("base_s2", "o-", "C0"), ("base_P0_s0", "s--", "C1"), ("base_P0_s1", "s--", "C1"),
                   ("base_P0_s2", "s--", "C1"), ("mixed_s0", "^-", "C2"), ("mixed_lam10", "v-", "C3"), ("wdown_k0.3", "d-", "C4"), ("feat0_s0", "*-", "C5"),
                   ("feat0_s1", "*-", "C5"), ("feat0_s2", "*-", "C5"), ("base_imp", "x:", "C6")]:
    x = [a for a in S["rows"] if a["name"] == n]
    if not x: continue
    x = x[0]; ok = [a not in x["collapsed"] for a in x["E"]]
    ax[0].plot(np.array(x["E"])[ok], np.array(x["loss"])[ok], st, color=col, ms=5, lw=1, label=DESC[n] if n.endswith("s0") or "_s" not in n else None)
    ax[1].plot(np.array(x["E"])[ok], np.array(x["E_var_nodes"])[ok], st, color=col, ms=5, lw=1)
ax[0].axvline(E1, color="k", ls=":"); ax[1].axhline(E1, color="k", ls=":", label="reference"); ax[1].plot([10.5, 14.5], [10.5, 14.5], color="0.7", lw=0.8, label="E_PINN = E")
ax[0].set_yscale("log"); ax[0].set_xlabel("E (node)"); ax[0].set_ylabel("variant loss (fresh sample)"); ax[0].legend(fontsize=6, frameon=False); ax[0].set_title("p = 3.5, disk: loss curves near $E_1$")
ax[1].set_xlabel("E (node)"); ax[1].set_ylabel("closed-form refined E of the trained network"); ax[1].set_ylim(11.2, 13.3); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("node dependence of the refined value")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "p35_curves.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
for n, (E, L, sd, lab, r) in fine.items():
    ax.errorbar(E, L, yerr=sd, marker="o", ms=4, lw=1.2, capsize=2, label=lab + " (bars: MC std, batch 5000)")
ax.axvline(E1, color="k", ls=":"); ax.set_yscale("log"); ax.set_xlabel("E"); ax.set_ylabel("variant loss"); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
ax.set_title("p = 3.5: fine scans (h = 0.25)"); fig.tight_layout(); fig.savefig(FIG / "p35_fine_curves.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, node in enumerate([10.5, 12.5]):
    for n, lab in [("base_s0", "baseline P1 (lr decay)"), ("base_P0c_s0", "baseline, constant lr"), ("base_P0_s0", "baseline P0 (early stop)"), ("mixed_s0", "mixed lam=100, P1"), ("feat0_s0", "singular feature, P1")]:
        d = load(n)
        if d is None: continue
        r = [x for x in d["records"] if abs(x["E"] - node) < 1e-9]
        if r:
            h = np.array(r[0]["hist"]); ax[k].plot(h[:, 0], h[:, 2 if k == 0 else 1], lw=0.9, label=lab)
    ax[k].set_yscale("log"); ax[k].set_xlabel("epoch"); ax[k].grid(alpha=0.3)
    ax[k].set_title(f"node E = {node}: " + ("total training loss (first node, from scratch)" if k == 0 else "residual term on the training batch (warm start)"))
ax[0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(FIG / "p35_training_histories.png", dpi=150); plt.close(fig)

# residual maps at E = 12.5
g = np.linspace(-1, 1, 201); X, Y = np.meshgrid(g, g); pts = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
maps = [("base_s0", "baseline: $(\\Delta_p u+E w)^2$"), ("wdown_k0.3", "down-weighted residual"), ("mixed_s0", "mixed: $(\\mathrm{div}\\,\\sigma+E w)^2$"),
        ("mixed_s0#c", "mixed: $100\\,|\\sigma-|\\nabla u|^{p-2}\\nabla u|^2$"), ("feat0_s0", "singular feature: $(\\Delta_p u+E w)^2$")]
fig, ax = plt.subplots(1, len(maps), figsize=(4 * len(maps), 3.6))
for a, (n, title) in zip(ax, maps):
    nm = n.split("#")[0]; d = load(nm); f = C.CKPT / f"{nm}_E12.500.pt"
    if d is None or not f.exists():
        a.set_axis_off(); continue
    job = d["config"]; model = C.make_model(job); model.load_state_dict(torch.load(f))
    vals = []
    for ch in torch.split(pts, 5000):
        T = C.terms(model, ch.clone(), job, False, False)
        vals.append((100.0 * T["aux"] if n.endswith("#c") else (T["a"] + 12.5 * T["b"]) ** 2).detach().flatten())
    Z = torch.cat(vals).numpy().reshape(X.shape); Z[X ** 2 + Y ** 2 > 1] = np.nan
    im = a.pcolormesh(X, Y, np.log10(Z + 1e-8), shading="auto", cmap="viridis", vmin=-4, vmax=2); a.set_aspect("equal"); a.set_title(title, fontsize=9)
    plt.colorbar(im, ax=a, fraction=0.046, label="log10 pointwise value")
fig.suptitle("p = 3.5, network trained at E = 12.5 (seed 0): pointwise loss density", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "p35_residual_maps.png", dpi=130); plt.close(fig)
