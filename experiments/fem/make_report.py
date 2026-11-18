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


# ------------------------------------------------------------------ PINN vs FEM rows
CASE_LABEL = {"double_well": "double well (3-D)", "four_well": "four well (3-D)", "harm_w10": r"harmonic $\omega=10$ (2-D)"}

def pinn_rows():
    rows = []
    for case in ["double_well", "four_well", "harm_w10"]:
        summ, res = PA.load_all(case)
        if not res: continue
        keys = sorted({k for s in res.values() for k, w in s["windows"].items() if w}, key=float)
        for k in keys:
            ws = [res[s]["windows"][k] for s in sorted(res) if k in res[s]["windows"] and res[s]["windows"][k]]
            ws = [w for w in ws if w["n"] >= 7]
            if not ws: continue
            rho = np.array([w["rho_res_at_min"] for w in ws]); par = np.array([w["E_par_r2"] for w in ws])
            parl = np.array([w["E_par_loss"] for w in ws]); nodem = np.array([w["node_min_loss"] for w in ws])
            fem_in = [r for r in summ if min(ws[0]["E"]) - 0.5 <= r["E"] <= max(ws[0]["E"]) + 0.5]
            est = float(np.mean(rho)); ref = min(fem_in, key=lambda r: abs(r["E"] - est))
            near = [r for r in fem_in if abs(r["E"] - ref["E"]) < 0.75]
            rows.append(dict(case=case, center=float(k), nseeds=len(ws), ref=ref["E"], ref_err=ref["err"], ref_mult=ref["mult"],
                             n_near=len(near), near=[(r["E"], r["mult"]) for r in near],
                             rho=rho.tolist(), par=par.tolist(), parl=parl.tolist(), nodem=nodem.tolist(),
                             t_E=float(np.mean([w["time_per_E"] for w in ws])), t_win=float(np.mean([w["time_total"] for w in ws])),
                             minr2=float(np.mean([w["min_r2"] for w in ws]))))
    return rows

def rel(a, b): return (a - b) / b

def pinn_markdown(rows):
    L = ["| case | window centre | FEM ref (mult) | #FEM eigenvalues within 0.75 | PINN Rayleigh quotient (mean over seeds [min,max]) | rel.err | parabola on normalised loss | rel.err | parabola on recorded loss (paper-style) | rel.err | min of Lambda | time/E-point |", "|" + "---|" * 12]
    for r in rows:
        f = lambda a: f"{np.nanmean(a):.4f} [{np.nanmin(a):.4f},{np.nanmax(a):.4f}]"
        L.append(f"| {r['case']} | {r['center']:.2f} | {r['ref']:.4f}±{r['ref_err']:.0e} ({r['ref_mult']}) | {r['n_near']} | {f(r['rho'])} | {100*rel(np.nanmean(r['rho']), r['ref']):+.3f}% | "
                 f"{f(r['par'])} | {100*rel(np.nanmean(r['par']), r['ref']):+.2f}% | {f(r['parl'])} | {100*rel(np.nanmean(r['parl']), r['ref']):+.2f}% | {r['minr2']:.3f} | {r['t_E']:.0f} s |")
    return "\n".join(L)

def pinn_tex(rows):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"example & window & FEM & mult. & $\rho$ (PINN) & rel.\ err.\ (\%) & parab.\ fit & rel.\ err.\ (\%) \\", r"\midrule"]
    last = None
    for r in rows:
        if last and r["case"] != last: L.append(r"\midrule")
        last = r["case"]
        rho = np.nanmean(r["rho"]); par = np.nanmean(r["par"])
        sd = np.nanstd(r["rho"]) if len(r["rho"]) > 1 else float("nan")
        flag = r"$^\dagger$" if r["n_near"] > 1 else ""
        L.append(f"{CASE_LABEL[r['case']]} & ${r['center']-1.5:.1f}$--${r['center']+1.5:.1f}$ & {r['ref']:.3f}{flag} & {r['ref_mult']} & {rho:.3f} & {100*rel(rho, r['ref']):+.2f} & {par:.2f} & {100*rel(par, r['ref']):+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)

def fem_tex(S):
    def block(case, title, ncol=1):
        rows = S[case]["rows"]
        return rows
    L = [r"\begin{tabular}{rrrrrr}", r"\toprule", r"\multicolumn{2}{c}{double well} & \multicolumn{2}{c}{four well} & \multicolumn{2}{c}{free ball (exact)} \\",
         r"$E$ & mult. & $E$ & mult. & $E$ & mult. \\", r"\midrule"]
    # list in order of the double-well, four-well, free-ball rows up to E<=40 and ground/shell clusters
    dw = [r for r in S["double_well"]["rows"] if r["E"] < 75]; fw = [r for r in S["four_well"]["rows"] if r["E"] < 75]; fb = [r for r in S["ball_free"]["rows"]]
    n = max(len(dw), len(fw), len(fb))
    cell = lambda rows, i: (f"{rows[i]['E']:.3f}", str(rows[i]["mult"])) if i < len(rows) else ("", "")
    for i in range(n):
        a, b, c = cell(dw, i), cell(fw, i), cell(fb, i)
        L.append(f"{a[0]} & {a[1]} & {b[0]} & {b[1]} & {c[0]} & {c[1]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)

def valid_tex(S):
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"problem & mult. & exact/reference & FEM (extrap.) & abs.\ err. & est.\ err. \\", r"\midrule"]
    spec = [("disk_free", "unit disk", 4), ("annulus", r"annulus $0.5<|x|<1$", 4), ("ball_free", "unit ball", 4), ("square", "unit square", 4), ("triangle", "triangle", 2),
            ("harm_w1", r"harmonic disk $\omega=1$", 4), ("harm_w10", r"harmonic disk $\omega=10$", 4)]
    for c, lab, n in spec:
        for j, r in enumerate(S[c]["rows"][:n]):
            L.append(f"{lab if j==0 else ''} & {r['exact_mult']} & {r['exact']:.6f} & {r['E']:.6f} & {abs(r['E']-r['exact']):.1e} & {r['err']:.1e} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def md_validation():
    L = ["| problem | mult | exact / reference | FEM extrapolated | abs. error | estimated error | obs. order p |", "|---|---|---|---|---|---|---|"]
    for c, lab, n in [("disk_free", "unit disk (j_mn^2)", 6), ("annulus", "annulus 0.5<r<1 (cross-product zeros)", 4), ("ball_free", "unit ball (j_{l+1/2,n}^2)", 7),
                      ("square", "unit square", 4), ("triangle", "right triangle", 2), ("harm_w1", "harmonic w=1 (radial FD ref.)", 6), ("harm_w10", "harmonic w=10 (radial FD ref.)", 6)]:
        for j, r in enumerate(S[c]["rows"][:n]):
            L.append(f"| {lab if j==0 else ''} | {r['exact_mult']} | {r['exact']:.6f} | {r['E']:.6f} | {abs(r['E']-r['exact']):.1e} | {r['err']:.1e} | {r['p_obs']:.2f} |")
    return "\n".join(L)

def md_levels(case, k=6):
    d = json.load(open(RES / f"fem_{case}.json"))["levels"]; ks = sorted(d, key=int)
    L = ["| level | dof | elements | wall (s) | " + " | ".join(f"E{j+1}" for j in range(k)) + " |", "|" + "---|" * (4 + k)]
    for kk in ks:
        e = d[kk]["eigenvalues"]; L.append(f"| {kk} | {d[kk]['ndof']} | {d[kk]['nelem']} | {d[kk]['wall']:.1f} | " + " | ".join(f"{x:.5f}" for x in e[:k]) + " |")
    return "\n".join(L)

def md_ref(case, emax):
    L = ["| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |", "|---|---|---|---|---|"]
    n = 0
    for r in S[case]["rows"]:
        if r["E"] > emax: break
        L.append(f"| {n+1}{'' if r['mult']==1 else '-'+str(n+r['mult'])} | {r['E']:.4f} | {r['err']:.1e} | {r['mult']} | {r['spread_fine']:.1e} |"); n += r["mult"]
    return "\n".join(L)

def write_md():
    out = ["<!-- generated by make_report.py -->", "### Validation", md_validation(), "", "### Mesh levels: ball, free", md_levels("ball_free"),
           "", "### Mesh levels: double well", md_levels("double_well"), "", "### Mesh levels: four well", md_levels("four_well"),
           "", "### Mesh levels: harmonic w=10", md_levels("harm_w10"), "", "### Mesh levels: annulus", md_levels("annulus", 5)]
    for c, e in [("double_well", 75), ("four_well", 75), ("harm_w1", 55), ("harm_w10", 55), ("annulus", 50), ("square", 105), ("triangle", 180)]:
        out += ["", f"### Reference eigenvalues: {c}, E<={e}", md_ref(c, e)]
    out += ["", "### PINN vs FEM", pinn_markdown(pinn_rows())]
    open(HERE / "results_tables.md", "w").write("\n".join(out))

if __name__ == "__main__":
    fig_spectra(); fig_convergence()
    for c, t in [("double_well", "Double well, 3-D ball"), ("four_well", "Four-well bump, 3-D ball"), ("harm_w10", r"Harmonic disk, $\omega=10$")]:
        fig_windows(c, t)
    rows = pinn_rows()
    json.dump(rows, open(RES / "fem_pinn_summary.json", "w"), indent=1)
    print(pinn_markdown(rows))
    open(HERE / "tables.tex", "w").write(
        "% generated by experiments/fem/make_report.py (booktabs; \\dagger: other FEM eigenvalues within 0.75 of the reference, i.e. unresolved cluster)\n"
        "% ---- Table: independent FEM validation\n" + valid_tex(S) + "\n\n% ---- Table: FEM reference eigenvalues\n" + fem_tex(S) +
        "\n\n% ---- Table: PINN vs FEM\n" + pinn_tex(rows) + "\n")
    write_md(); print("done")
