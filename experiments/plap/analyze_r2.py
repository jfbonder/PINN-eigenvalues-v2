"""Round 2: seed statistics, E2 convergence row, node dependence for p=3,4.  Writes tables_r2.md and tables_r2.tex.
Run: python -m experiments.plap.analyze_r2"""
import json
from pathlib import Path
import numpy as np
from experiments.plap.analyze import curve, vertex3
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
S = json.load(open(RES / "plap_summary.json")); refs = json.load(open(RES / "plap_refs.json"))
fem = json.load(open(RES / "plap_fem_refs.json"))
pc = lambda x: f"{100*x:+.2f}"


def one(name, ref, i=None):
    c = curve(name)
    if c is None or len(c[0]) < 3: return None
    E, L, r, d = c
    ev, i, _ = vertex3(E, L, i)
    return dict(name=name, seed=d["config"].get("seed", 0), E_node=float(E[i]), E_ls=r[i]["E_ls"], E_vertex=ev, Lam=float(L[i]),
                core=r[i]["core_frac"], mass_nodes=[x["mass"] for x in r], E_ls_nodes=[x["E_ls"] for x in r], nodes=E.tolist(), Lam_nodes=L.tolist(),
                wall=sum(x["wall_time"] for x in r), ref=ref)


def stats(rows, key, ref):
    v = np.array([x[key] for x in rows]); return v.mean(), v.max() - v.min(), (v.mean() - ref) / ref, np.max(np.abs(v - ref)) / ref

md, tex, out = [], [], {}
# ---- (3) seeds for main results ----
G = [("disk $E_1$", "disk E1", ["conv_base_s0", "conv_base_s1", "conv_base_s2"], refs["p2.2_d2"]["E1"], None),
     ("disk $E_2$", "disk E2", ["disk22_loc1", "r2_diskE2_s1", "r2_diskE2_s2"], fem["halfdisk_p2.2"]["extrap"], None),
     ("square $E_1$", "square E1", ["square22_loc0", "r2_sqE1_s1", "r2_sqE1_s2"], fem["square2_p2.2"]["extrap"], None),
     ("square $E_2$", "square E2", ["square22_loc1", "r2_sqE2_s1", "r2_sqE2_s2"], None, None)]
md.append("### Round 2 (a): three seeds, local windows, corrected loss, protocol P1 (p = 2.2)\n\n| case | ref | E_PINN per seed (0,1,2) | mean | spread (max-min) | err of mean % | max abs err % | E_curve per seed | Lam_min per seed |\n|---|---|---|---|---|---|---|---|---|")
tex += [r"% ---------------- Round 2 ----------------", r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Eigenvalue & $E_{\rm ref}$ & $E_{\rm PINN}$ (seeds 0, 1, 2) & mean & spread & max.\ $|$err.$|$ (\%) & $E_{\rm curve}$ (seeds 0, 1, 2) \\", r"\midrule"]
for lt, lm, names, ref, _ in G:
    if ref is None:   # square E2: each seed is compared with the nearer of the two close eigenvalues
        cands = [fem["tri_diag_p2.2"]["extrap"], fem["rect1x2_p2.2"]["extrap"]]
        rows = []
        for n in names:
            r0 = one(n, cands[0])
            if r0: r0["ref"] = min(cands, key=lambda a: abs(a - r0["E_ls"])); rows.append(r0)
        if not rows: continue
        errs = [abs(x["E_ls"] - x["ref"]) / x["ref"] for x in rows]
        md.append(f"| {lm} (two close eigenvalues 15.1429 diag / 15.2808 midline) | nearest | " + ", ".join(f"{x['E_ls']:.4f}→{x['ref']:.4f}" for x in rows)
                  + f" | -- | -- | -- | {100*max(errs):.2f} | " + ", ".join(f"{x['E_vertex']:.3f}" for x in rows) + " | " + ", ".join(f"{x['Lam']:.3f}" for x in rows) + " |")
        tex.append(f"{lt} & 15.1429 / 15.2808 & " + ", ".join(f"{x['E_ls']:.4f}" for x in rows) + f" & -- & -- & {100*max(errs):.2f} & "
                   + ", ".join(f"{x['E_vertex']:.2f}" for x in rows) + r" \\")
        out[lm] = rows; continue
    rows = [x for x in (one(n, ref) for n in names) if x]
    if not rows: continue
    m, sp, em, mx = stats(rows, "E_ls", ref)
    md.append(f"| {lm} | {ref:.4f} | " + ", ".join(f"{x['E_ls']:.4f}" for x in rows) + f" | {m:.4f} | {sp:.4f} | {pc(em)} | {100*mx:.2f} | "
              + ", ".join(f"{x['E_vertex']:.3f}" for x in rows) + " | " + ", ".join(f"{x['Lam']:.3f}" for x in rows) + " |")
    tex.append(f"{lt} & {ref:.4f} & " + ", ".join(f"{x['E_ls']:.4f}" for x in rows) + f" & {m:.4f} & {sp:.4f} & {100*mx:.2f} & "
               + ", ".join(f"{x['E_vertex']:.2f}" for x in rows) + r" \\")
    out[lm] = rows
tex += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Seed dependence of the corrected-loss results for $p=2.2$ (three seeds, local three-node scans with $h=0.5$, baseline settings). "
        r"For the second eigenvalue of the square the two close eigenvalues $15.1429$ (diagonal nodal line) and $15.2808$ (midline) lie in the same minimum of the loss curve; "
        r"each seed converges to one of them and the error is measured to the nearer one.}", r"\label{tab.plap.seeds}", r"\end{table}", ""]

# ---- (2) convergence row(s) for E2 of the disk ----
ref2 = fem["halfdisk_p2.2"]["extrap"]
C = [("disk22_loc1", "baseline, seed 0"), ("r2_diskE2_s1", "baseline, seed 1"), ("r2_diskE2_s2", "baseline, seed 2"),
     ("r2_diskE2_P0", "P0 (constant lr, early stop)"), ("r2_diskE2_w64d2", r"network 2x64"), ("r2_diskE2_ep5000", "5000 epochs")]
md.append(f"\n### Round 2 (b): convergence for the SECOND eigenvalue of the disk, p = 2.2 (ref {ref2:.4f}; nodes 18.0, 18.5, 19.0)\n\n| setting | Lam(18.0) | Lam(18.5) | Lam(19.0) | core frac | E_curve | err % | E_PINN (node 18.5) | err % | E_PINN over the 3 nodes | wall min |\n|---|---|---|---|---|---|---|---|---|---|---|")
tex += [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lccccc}", r"\toprule",
        r"Setting & $\bar\Lambda_p(18.0)$ & $\bar\Lambda_p(18.5)$ & $\bar\Lambda_p(19.0)$ & $E_{\rm curve}$ (err.\ \%) & $E_{\rm PINN}$ (err.\ \%) \\", r"\midrule"]
rows2 = []
for n, dsc in C:
    x = one(n, ref2, 1)
    if not x: continue
    x["desc"] = dsc; rows2.append(x)
    L = x["Lam_nodes"]
    md.append(f"| {dsc} | {L[0]:.3f} | {L[1]:.3f} | {L[2]:.3f} | {x['core']:.2f} | {x['E_vertex']:.3f} | {pc((x['E_vertex']-ref2)/ref2)} | {x['E_ls']:.4f} | {pc((x['E_ls']-ref2)/ref2)} | "
              f"{min(x['E_ls_nodes']):.4f}–{max(x['E_ls_nodes']):.4f} | {x['wall']/60:.0f} |")
    tex.append(f"{dsc.replace('2x64', r'$2\\times64$')} & {L[0]:.3f} & {L[1]:.3f} & {L[2]:.3f} & {x['E_vertex']:.2f} (${pc((x['E_vertex']-ref2)/ref2)}$) & {x['E_ls']:.4f} (${pc((x['E_ls']-ref2)/ref2)}$) \\\\")
tex += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Convergence study for the second eigenvalue of the $p$-Laplacian, $p=2.2$, in the unit disk (reference $%.4f$: finite elements on the half disk; $18.395$ in \cite{HJ11}). Nodes $E=18.0,18.5,19.0$; $E_{\rm PINN}$ from the network trained at $E=18.5$.}".replace("%.4f", f"{ref2:.4f}"),
        r"\label{tab.plap.conv2}", r"\end{table}", ""]
out["E2_conv"] = rows2

# ---- (1) other p: seeds and node dependence ----
md.append("\n### Round 2 (c): p = 3 and p = 4, unit disk, E1: three seeds and node dependence of E_PINN\n\n| p | seed | ref | nodes | Lam at nodes | E_PINN at nodes | grid-min node | E_PINN there (err %) | E_curve (err %) | node range of E_PINN, 3 central nodes (err %) |\n|---|---|---|---|---|---|---|---|---|---|")
tex += [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{ccccc}", r"\toprule",
        r"$p$ & $E_1$ (shooting) & $E_{\rm PINN}$ at the grid minimum (seeds 0, 1, 2) & max.\ $|$err.$|$ (\%) & range of $E_{\rm PINN}$ over nodes and seeds (err.\ \%) \\", r"\midrule"]
tex.append(f"2.2 & {refs['p2.2_d2']['E1']:.4f} & " + ", ".join(f"{x['E_ls']:.4f}" for x in out.get('disk E1', [])) + " & "
           + (f"{100*max(abs(x['E_ls']-x['ref'])/x['ref'] for x in out['disk E1']):.2f}" if out.get('disk E1') else "--") + " & "
           + (lambda v, r: f"{min(v):.3f}--{max(v):.3f} (${pc((min(v)-r)/r)}$ to ${pc((max(v)-r)/r)}$)")([e for x in out.get('disk E1', []) for e in x['E_ls_nodes'][-3:]] or [0], refs['p2.2_d2']['E1']) + r" \\")
for p, names in [(3.0, ["disk_p3", "r2_disk_p3_s1", "r2_disk_p3_s2"]), (4.0, ["disk_p4", "r2_disk_p4_s1", "r2_disk_p4_s2"])]:
    ref = refs[f"p{p}_d2"]["E1"]; rows = [x for x in (one(n, ref) for n in names) if x and len(x["nodes"]) == 5]
    allc = []; ncoll = 0
    for x in rows:
        coll = [x["nodes"][j] for j in range(5) if x["mass_nodes"][j] < 0.5]
        ncoll += len(coll)
        cen = [x["E_ls_nodes"][j] for j in range(1, 4) if x["mass_nodes"][j] >= 0.5]; allc += cen
        x["collapsed_nodes"] = coll
        md.append(f"| {p} | {x['seed']} | {ref:.4f} | {x['nodes']} | " + ", ".join(f"{v:.2f}" for v in x["Lam_nodes"]) + " | " + ", ".join(f"{v:.3f}" for v in x["E_ls_nodes"])
                  + f" | {x['E_node']} | {x['E_ls']:.4f} ({pc((x['E_ls']-ref)/ref)}) | {x['E_vertex']:.3f} ({pc((x['E_vertex']-ref)/ref)}) | {min(cen):.3f}–{max(cen):.3f} ({pc((min(cen)-ref)/ref)} to {pc((max(cen)-ref)/ref)})" + (f"; COLLAPSED to u=0 at E={coll}" if coll else "") + " |")
    if rows:
        mx = max(abs(x["E_ls"] - ref) / ref for x in rows)
        md.append(f"| {p} | all | | | | | | max abs err {100*mx:.2f} % | | {min(allc):.3f}–{max(allc):.3f} ({pc((min(allc)-ref)/ref)} to {pc((max(allc)-ref)/ref)}) |")
        tex.append(f"{p:g} & {ref:.4f} & " + ", ".join(f"{x['E_ls']:.3f}" for x in rows) + f" & {100*mx:.2f} & {min(allc):.2f}--{max(allc):.2f} (${pc((min(allc)-ref)/ref)}$ to ${pc((max(allc)-ref)/ref)}$)"
                   + (f"; {ncoll} of {5*len(rows)} trainings collapsed to $u\\equiv0$" if ncoll else "") + r" \\")
    out[f"p{p}"] = rows
tex += [r"\bottomrule", r"\end{tabular}",
        r"\caption{First eigenvalue of the $p$-Laplacian in the unit disk for $p=2.2,3,4$: three seeds, and dependence of $E_{\rm PINN}$ on the node $E$ at which the network was trained "
        r"(three central nodes of the local grid, all seeds; step $h=0.5$, $0.75$, $1$ for $p=2.2,3,4$).}", r"\label{tab.plap.p.seeds}", r"\end{table}", ""]

# ---- legacy near the published second value ----
c = curve("r2_disk22_legacyE2_P1", "Lam")
if c is not None and len(c[0]) >= 3:
    E, L, r, d = c; ev, i, _ = vertex3(E, L)
    md.append(f"\n### Round 2 (d): legacy loss near the published second value of the disk (19.44; true 18.393)\n\nnodes {E.tolist()}, legacy loss {np.round(L,3).tolist()}, "
              f"frozen-network E {[round(x['E_ls'],3) for x in r]}; grid minimum {E[i]}, 3-pt vertex {ev:.3f}.")
    out["legacy_E2"] = dict(nodes=E.tolist(), Lam=L.tolist(), E_ls=[x["E_ls"] for x in r], vertex=ev)
json.dump(out, open(RES / "plap_summary_r2.json", "w"), indent=1)
open(HERE / "tables_r2.md", "w").write("\n".join(md) + "\n"); open(HERE / "tables_r2.tex", "w").write("\n".join(tex) + "\n")
print("\n".join(md))
