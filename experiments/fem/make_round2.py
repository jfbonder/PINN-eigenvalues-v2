"""Round-2 tables: per-seed worst case, blind scan vs FEM, extended harmonic window, Tables 2-4 re-runs, MC standard errors."""
import json, glob
from functools import partial
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
S = json.load(open(RES / "fem_summary.json"))
pct = lambda a, b: 100 * (a - b) / b

def parab(E, y, i):
    if i == 0 or i == len(E) - 1: return float("nan")
    h = E[i + 1] - E[i]; d = y[i - 1] - 2 * y[i] + y[i + 1]
    return float(E[i] + 0.5 * h * (y[i - 1] - y[i + 1]) / d) if d > 0 else float("nan")

def per_seed_md():
    rows = json.load(open(RES / "fem_pinn_summary.json"))
    L = ["| example | window | FEM ± est.err (mult) | other FEM eigenvalues within 0.75 | rho per seed | rel.err per seed (%) | worst |", "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule", r"example & window & FEM $\pm$ unc. & mult. & $\rho$ per seed & worst rel.\ err.\ (\%) \\", r"\midrule"]
    worst = 0
    for r in rows:
        er = [pct(x, r["ref"]) for x in r["rho"]]; w = max(abs(e) for e in er); worst = max(worst, w)
        others = ", ".join(f"{e:.3f}(x{m})" for e, m in r["near"] if abs(e - r["ref"]) > 1e-9) or "-"
        L.append(f"| {r['case']} | {r['center']-1.5:.1f}-{r['center']+1.5:.1f} | {r['ref']:.4f} ± {r['ref_err']:.0e} ({r['ref_mult']}) | {others} | {', '.join(f'{x:.4f}' for x in r['rho'])} | {', '.join(f'{e:+.3f}' for e in er)} | {w:.3f} |")
        tex.append(f"{r['case'].replace('_',' ')} & {r['center']-1.5:.1f}--{r['center']+1.5:.1f} & ${r['ref']:.4f}\\pm{r['ref_err']:.0e}$ & {r['ref_mult']} & {', '.join(f'{x:.3f}' for x in r['rho'])} & {w:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L), "\n".join(tex), worst

def scan_md():
    p = RES / "fem_r2_scan_s0.json"
    if not p.exists(): return "", "", None
    recs = sorted(json.load(open(p))["windows"]["scan_4.7-49.7"], key=lambda r: r["E"])
    E = np.array([r["E"] for r in recs]); lr = np.array([r["loss_res"] for r in recs]); r2 = np.array([r["r2"] for r in recs]); rho = np.array([r["rho_res"] for r in recs])
    fem = S["double_well"]["rows"]; femE = [(r["E"], r["mult"]) for r in fem if r["E"] < 52]
    def mins(y): return [i for i in range(1, len(E) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]
    mi = mins(lr)
    L = [f"Blind scan, double well, nodes E=4.7..49.7 step 1.5 ({len(E)} nodes done), mu=400, 1 seed. Local minima of the recorded loss (paper-style detection):", "",
         "| min at node E | recorded loss | normalised loss r^2 | parab. fit (recorded) | Rayleigh quotient at node | FEM eigenvalues within 1.5 (mult) | nearest FEM | err of rho |", "|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{rrrrl}", r"\toprule", r"min.\ node & parab.\ fit & $\rho$ (net) & FEM within 1.5 (mult.) & err.\ of $\rho$ (\%) \\", r"\midrule"]
    found = []
    for i in mi:
        near = [(e, m) for e, m in femE if abs(e - E[i]) <= 1.5]
        pf = parab(E, lr, i); nn = min(femE, key=lambda t: abs(t[0] - rho[i]))
        L.append(f"| {E[i]:.1f} | {lr[i]:.3f} | {r2[i]:.3f} | {pf:.3f} | {rho[i]:.4f} | {', '.join(f'{e:.3f}(x{m})' for e,m in near) or '-'} | {nn[0]:.4f} | {pct(rho[i], nn[0]):+.3f}% |")
        tex.append(f"{E[i]:.1f} & {pf:.2f} & {rho[i]:.3f} & {', '.join(f'{e:.2f}({m})' for e,m in near) or '--'} & {pct(rho[i], nn[0]):+.2f} \\\\")
        found.append(pf if np.isfinite(pf) else E[i])
    tex += [r"\bottomrule", r"\end{tabular}"]
    miss = [(e, m) for e, m in femE if all(abs(e - f) > 0.75 for f in found)]
    L += ["", "FEM eigenvalue groups (E<52) with NO detected minimum within 0.75 (distinct eigenvalues listed individually; clusters = several within a few tenths): " + (", ".join(f"{e:.3f}(x{m})" for e, m in miss) or "none")]
    L += ["", "Node values: " + "; ".join(f"{e:.1f}: loss {a:.2f}, rho {b:.3f}" for e, a, b in zip(E, lr, rho))]
    return "\n".join(L), "\n".join(tex), None

def ext_md():
    p = RES / "fem_r2_ext_s0.json"
    if not p.exists(): return ""
    recs = sorted(json.load(open(p))["windows"]["ext_45.2-49.2"], key=lambda r: r["E"])
    L = ["| E | recorded loss | normalised r^2 | rho (net) ± SE | r | mass |", "|---|---|---|---|---|---|"]
    for r in recs: L.append(f"| {r['E']:.1f} | {r['loss_res']:.3f} | {r['r2']:.3f} | {r['rho_res']:.4f} ± {r['rho_res_se']:.4f} | {r['r']:.3f} | {r['mass']:.3f} |")
    return "\n".join(L)

PAPER = {  # v4 Tables 2-4: exact -> (paper our method, rel err)
 "ball3": {9.8696: 9.885022, 20.1907: 20.182569, 33.2175: 33.243410, 39.4784: 39.494155, 48.8312: 48.826976, 59.6795: 59.734069, 66.9543: 66.991853},
 "ball4": {14.6820: 14.641796, 26.3746: 26.257056, 40.7065: 40.794700, 49.2185: 49.175395},
 "tri": {5 * np.pi**2: 49.369942, 10 * np.pi**2: 98.892161, 13 * np.pi**2: 128.048509, 17 * np.pi**2: 167.687185}}
LAB = {"ball3": "Table 2 (3-D ball, mu=100)", "ball4": "Table 3 (4-D ball, mu=200)", "tri": "Table 4 (triangle, mu=15000)"}
EXACT = {"ball3": [9.869604, 20.190729, 33.217462, 39.478418, 48.831194, 59.679516, 66.954312],
         "ball4": [14.681971, 26.374616, 40.706466, 49.218456], "tri": [5 * np.pi**2, 10 * np.pi**2, 13 * np.pi**2, 17 * np.pi**2]}

def tables_md():
    md, tex = [], []
    for t in ["tri", "ball3", "ball4"]:
        p = RES / f"fem_r2_{t}_s0.json"
        if not p.exists(): continue
        d = json.load(open(p))["windows"]
        md += [f"**{LAB[t]}**: 3 offset nodes (exact-0.9, +0.1, +1.1) per eigenvalue, fresh network at the first node, warm start along the window; rho = closed-form frozen-network Rayleigh quotient (4e5 MC pts, standard error in brackets).", "",
               "| exact | rho (min-loss node) ± SE | rel.err | parabola on r^2 | rel.err | r ± SE | mass | min r^2 | paper value (v4) | paper rel.err |", "|---|---|---|---|---|---|---|---|---|---|"]
        tex += [rf"\multicolumn{{7}}{{l}}{{{LAB[t]}}} \\"]
        for (k, recs), ex in zip(d.items(), EXACT[t]):
            recs = sorted(recs, key=lambda r: r["E"])
            if len(recs) < 3: continue
            E = np.array([r["E"] for r in recs]); r2 = np.array([r["r2"] for r in recs]); i = int(np.argmin(r2)); r = recs[i]
            pf = parab(E, r2, i); key = min(PAPER[t], key=lambda z: abs(z - ex)); pv = PAPER[t][key]
            md.append(f"| {ex:.4f} | {r['rho_res']:.4f} ({r['rho_res_se']:.4f}) | {pct(r['rho_res'], ex):+.4f}% | {pf:.3f} | {pct(pf, ex):+.3f}% | {r['r']:.3f} ± {r['r_se']:.3f} | {r['mass']:.3f} | {r['r2']:.3f} | {pv:.4f} | {pct(pv, ex):+.4f}% |")
            tex.append(f"{ex:.4f} & {r['rho_res']:.4f} & {pct(r['rho_res'], ex):+.3f} & {pf:.3f} & {r['r']:.2f} & {r['mass']:.2f} & {pct(pv, ex):+.3f} \\\\")
        md += ["", "All nodes (E: Rayleigh quotient, r, epochs; a node with r >> 1 or rho far from E has NOT converged to the targeted eigenfunction - cold start at the first node of each window):", ""]
        for (k, recs) in d.items():
            md.append(f"- window {k}: " + "; ".join(f"{r['E']:.2f}: rho={r['rho_res']:.3f}, r={r['r']:.2f}, ep={r['epochs_run']}" for r in sorted(recs, key=lambda r: r['E'])))
        md.append("")
        tex.append(r"\addlinespace")
    return "\n".join(md), "\n".join(tex)

def mc_se_md():
    from pinn_eigen import EigenFunctionNN
    import pinn_eigen as pe
    from experiments.fem.mc_eval import mc_eval
    torch.set_num_threads(1)
    out = RES / "fem_mc_se.json"
    if out.exists(): data = json.load(open(out))
    else:
        data = []
        specs = [("harm_w10_s0_E45.200", 2, pe.boundary_disk, partial(pe.V_harmonic, omega=10.), partial(pe.sample_disk, radius=1.), 45.2),
                 ("harm_w10_s0_E14.280", 2, pe.boundary_disk, partial(pe.V_harmonic, omega=10.), partial(pe.sample_disk, radius=1.), 14.28),
                 ("four_well_s0_E9.850", 3, pe.boundary_ball, pe.V_four_well_bump, pe.sample_ball_3d, 9.85),
                 ("four_well_s0_E39.470", 3, pe.boundary_ball, pe.V_four_well_bump, pe.sample_ball_3d, 39.47),
                 ("double_well_s1_E19.860", 3, pe.boundary_ball, partial(pe.V_double_well_gaussian, a=1., sigma=.5), pe.sample_ball_3d, 19.86),
                 ("double_well_s1_E39.200", 3, pe.boundary_ball, partial(pe.V_double_well_gaussian, a=1., sigma=.5), pe.sample_ball_3d, 39.2)]
        for name, dim, B, V, samp, E in specs:
            f = RES / "fem_pinn_models" / f"{name}.pt"
            if not f.exists(): continue
            m = EigenFunctionNN(dim, B); m.load_state_dict(torch.load(f))
            data.append(dict(name=name, **mc_eval(m, samp, V, E)))
        json.dump(data, open(out, "w"), indent=1)
    L = ["| run (saved network) | rho_res ± SE | rho_var ± SE | r ± SE | mean u^2 ± SE |", "|---|---|---|---|---|"]
    for d in data:
        L.append(f"| {d['name']} | {d['rho_res']:.4f} ± {d['rho_res_se']:.4f} | {d['rho_var']:.3f} ± {d['rho_var_se']:.3f} | {d['r']:.3f} ± {d['r_se']:.3f} | {d['mass']:.3f} ± {d['mass_se']:.3f} |")
    return "\n".join(L)

if __name__ == "__main__":
    ps, ps_tex, worst = per_seed_md(); sc, sc_tex, _ = scan_md(); ex = ext_md(); tmd, ttex = tables_md(); se = mc_se_md()
    md = ["## 8. Round 2 (referee H: items 4, 5, 10, 15)", "", "### 8.1 Per-seed worst case and FEM uncertainty (item 4)", ps, "",
          f"Worst single-run error of the Rayleigh quotient over all windows/seeds: {worst:.3f}%. (Seed means are within 0.1%, single runs within 0.25%; the 0.25% case is double well 32.9, a window containing 3 FEM eigenvalues with spread 0.25, i.e. inside the cluster.) The FEM uncertainty (Richardson estimate, conservative) is 5e-4..2e-2 in 3-D and exceeds some of the PINN errors; for clusters the reference is the cluster, not a single member.", "",
          "### 8.2 Blind scan of the double-well loss curve vs FEM (item 4)", sc, "", "### 8.3 Harmonic w=10, window extended to 49.2 (item 5)", ex, "",
          "### 8.4 Tables 2-4 re-run with local offset windows (item 10)", tmd, "### 8.5 Monte Carlo standard errors (item 15)", se]
    (HERE / "README_round2.md").write_text("\n".join(md))
    (HERE / "tables_round2.tex").write_text("% Round-2 tables (generated by make_round2.py)\n% per-seed worst case\n" + ps_tex + "\n\n% blind scan minima vs FEM\n" + sc_tex +
        "\n\n% Tables 2-4 re-run\n\\begin{tabular}{rrrrrrr}\n\\toprule\nexact & $\\rho$ & rel.\\ err.\\ (\\%) & parab. & $r$ & mass & paper rel.\\ err.\\ (\\%) \\\\\n\\midrule\n" + ttex + "\n\\bottomrule\n\\end{tabular}\n")
    print("ok")
