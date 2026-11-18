"""Collect results -> experiments/results/plap_summary.json, figures, Markdown tables (stdout) and tables.tex.
Run: python -m experiments.plap.analyze"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.plap import jobs

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)


def load(name):
    f = RES / f"plap_{name}.json"
    return json.load(open(f)) if f.exists() else None


def curve(name, key="Lam_n"):
    d = load(name)
    if d is None or not d["records"]:
        return None
    r = d["records"]
    return np.array([x["E"] for x in r]), np.array([x[key] for x in r]), r, d


def vertex3(E, L, i=None):
    if i is None:
        i = int(np.argmin(L))
    i = min(max(i, 1), len(E) - 2)
    c = np.polyfit(E[i - 1:i + 2], L[i - 1:i + 2], 2)
    return float(-c[1] / (2 * c[0])), i, float(c[0])


def minima(E, L):
    idx = [i for i in range(1, len(L) - 1) if L[i] < L[i - 1] and L[i] < L[i + 1]]
    return [i for i in idx if L[i] < 0.5 * np.median(L)]


def est(name, ref=None, i=None):
    """Estimators from a (local) scan: 3-point vertex, grid minimum, E_ls and RQ at the grid minimum."""
    c = curve(name)
    if c is None or len(c[0]) < 3:
        return None
    E, L, r, d = c
    ev, i, curv = vertex3(E, L, i)
    out = dict(name=name, h=float(E[1] - E[0]), E_grid=float(E[i]), E_vertex=ev, curv=curv, Lam_min=float(L[i]),
               r_min=float(np.sqrt(L[i])), E_ls=r[i]["E_ls"], RQ=r[i]["RQ"], core_frac=r[i]["core_frac"],
               train_res=r[i]["train_res"], wall_s=sum(x["wall_time"] for x in r), npts=len(E),
               epochs=[x["epochs_run"] for x in r])
    if ref:
        out["ref"] = ref
        for k in ("E_grid", "E_vertex", "E_ls", "RQ"):
            out["err_" + k] = (out[k] - ref) / ref
    return out


def pct(x, nd=2):
    return "--" if x is None else f"{100 * x:+.{nd}f}"


def main():
    refs = json.load(open(RES / "plap_refs.json"))
    fem = json.load(open(RES / "plap_fem_refs.json")) if (RES / "plap_fem_refs.json").exists() else {}
    fx = lambda k: fem.get(k, {}).get("extrap")
    R = dict(disk22_E1=refs["p2.2_d2"]["E1"], disk22_E2=fx("halfdisk_p2.2") or 18.395,
             sq_E1=fx("square2_p2.2") or 5.4952,
             sq_E2=min([v for v in (fx("rect1x2_p2.2"), fx("tri_diag_p2.2")) if v] or [15.144]),
             ball_E1=refs["p2.5_d3"]["E1"], ball_E2=fx("halfball3_axi_p2.5"))
    S = dict(refs=R, fem={k: dict(extrap=v["extrap"], finest=v["finest"], order=v["order"]) for k, v in fem.items()})
    md = []

    # ---------- references ----------
    md.append("### References\n\n| p | d | E1(B_1) shooting | 2nd radial |\n|---|---|---|---|")
    for k, v in refs.items():
        if k.startswith("p"):
            md.append(f"| {v['p']} | {v['d']} | {v['E1']:.6f} | {v['E_rad2']:.4f} |")
    md.append("\n| FEM (P1, Richardson) | p | finest mesh | extrapolated | order |\n|---|---|---|---|---|")
    for k, v in fem.items():
        md.append(f"| {v['domain']} | {v['p']} | {v['finest']:.5f} | {v['extrap']:.5f} | {v['order']:.2f} |")

    # ---------- corrected re-runs: every detected minimum of the (merged) coarse curves, local 3-point scans ----
    cross = fx("tri_diag_p2.2") * 2 ** 1.1 if fx("tri_diag_p2.2") else None
    REFS = dict(
        disk=[(refs["p2.2_d2"]["E1"], "radial shooting (1st eigenvalue)"), (fx("halfdisk_p2.2"), "FEM half disk (1 nodal diameter); Horak 18.395"),
              (fx("sector_pi/2_p2.2"), "FEM sector pi/2 (2 nodal diameters)"), (refs["p2.2_d2"]["E_rad2"], "radial shooting, 1 nodal circle"),
              (fx("sector_pi/3_p2.2"), "FEM sector pi/3 (3 nodal diameters)")],
        square=[(fx("square2_p2.2"), "FEM square (1st eigenvalue); Horak 5.4952"), (fx("tri_diag_p2.2"), "FEM half square, diagonal nodal line; Horak 15.144"),
                (fx("rect1x2_p2.2"), "FEM rectangle 1x2 (midline nodal line)"), (fx("unit_square_p2.2"), "FEM [0,1]^2 = 2^p E1 (2x2 cells)"),
                (cross, "scaled FEM triangle (both diagonals nodal), 2^{p/2} x 15.1429"), (fx("strip3_p2.2"), "FEM strip 2/3 x 2 (3 strips)")])
    S["mode_refs"] = {k: [dict(E=a, type=b) for a, b in v] for k, v in REFS.items()}
    rows = []
    for dom, base, window, coarse in [("disk", "disk22_loc", "[3,58]", ["disk22_coarse_P1", "disk22_coarse2_P1"]),
                                      ("square", "square22_loc", "[3,37]", ["square22_coarse_P1", "square22_coarse2_P1"])]:
        for k in range(12):
            e = est(f"{base}{k}")
            if not e: continue
            cand = [(a, b) for a, b in REFS[dom] if a]
            j = int(np.argmin([abs(e["E_ls"] - a) for a, _ in cand]))
            e.update(label=f"{dom} p=2.2, window {window}, minimum {k+1}", dom=dom, k=k + 1, ref=cand[j][0], ref_type=cand[j][1],
                     coarse_vertex=load(f"{base}{k}")["config"].get("coarse_vertex"))
            for q in ("E_grid", "E_vertex", "E_ls", "RQ"):
                e["err_" + q] = (e[q] - e["ref"]) / e["ref"]
            e["err_coarse_vertex"] = (e["coarse_vertex"] - e["ref"]) / e["ref"]
            rows.append(e)
    for k, (nm, ref, typ) in enumerate([("ball25_E1_local", R["ball_E1"], "radial shooting (1st eigenvalue)"),
                                        ("ball25_E2_local", R["ball_E2"], "FEM axisymmetric half ball (nodal plane)")]):
        e = est(nm, ref)
        if e:
            e.update(label=f"ball3 p=2.5, minimum {k+1}", dom="ball3", k=k + 1, ref_type=typ,
                     coarse_vertex=load(nm)["config"].get("coarse_vertex"))
            if ref: e["err_coarse_vertex"] = (e["coarse_vertex"] - ref) / ref
            rows.append(e)
    S["main"] = rows
    md.append("\n### Corrected loss (protocol P1): detected eigenvalues\n\n| case | reference (type) | coarse 3-pt vertex | err % | local 3-pt vertex | err % | E_ls | err % | "
              "Lam_min (fresh) | r=sqrt | core frac |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for e in rows:
        ref = e.get("ref")
        md.append(f"| {e['label']} | {'--' if ref is None else f'{ref:.4f}'} ({e['ref_type']}) | {e['coarse_vertex']:.3f} | {pct(e.get('err_coarse_vertex'))} | "
                  f"{e['E_vertex']:.3f} | {pct(e.get('err_E_vertex'))} | "
                  f"{e['E_ls']:.4f} | {pct(e.get('err_E_ls'))} | "
                  f"{e['Lam_min']:.3f} | {e['r_min']:.3f} | {e['core_frac']:.2f} |")

    # ---------- paper protocol / legacy ----------
    rows2 = []
    for label, name, ref, pub in [("disk p=2.2 correct loss, P0 (paper protocol)", "disk22_coarse_P0", None, None),
                                  ("square p=2.2 correct loss, P0", "square22_coarse_P0", None, None)]:
        c = curve(name)
        if c is None: continue
        E, L, r, d = c
        refs_k = [R["disk22_E1"], R["disk22_E2"]] if "disk" in name else [R["sq_E1"], R["sq_E2"]]
        mins = [i for i in range(1, len(L) - 1) if L[i] < L[i - 1] and L[i] < L[i + 1]]
        for i in mins:
            ev, _, _ = vertex3(E, L, i)
            k = int(np.argmin([abs(ev - x) for x in refs_k]))
            rows2.append(dict(label=label, E_grid=float(E[i]), E_vertex=ev, Lam=float(L[i]), train_res=r[i]["train_res"],
                              E_ls=r[i]["E_ls"], nearest_ref=refs_k[k], err=(ev - refs_k[k]) / refs_k[k]))
    for label, name, pub, true in [("disk p=2.2 LEGACY loss, P0", "disk22_legacy_P0", 6.679, R["disk22_E1"]),
                                   ("disk p=2.2 LEGACY loss, P1", "disk22_legacy_P1", 6.679, R["disk22_E1"]),
                                   ("ball3 p=2.5 LEGACY loss, P1", "ball25_legacy_P1", 17.4513, R["ball_E1"])]:
        c = curve(name, "Lam")
        if c is None or len(c[0]) < 3: continue
        E, L, r, d = c
        ev, i, _ = vertex3(E, L)
        rows2.append(dict(label=label, E_grid=float(E[i]), E_vertex=ev, Lam=float(L[i]), train_res=r[i]["train_res"],
                          E_ls=r[i]["E_ls"], published=pub, nearest_ref=true, err=(ev - true) / true))
    S["paper_protocol_and_legacy"] = rows2
    md.append("\n### Paper protocol (P0) and legacy loss\n\n| run | grid min | 3-pt vertex | E_ls | Lam at min | published | true | err % |\n|---|---|---|---|---|---|---|---|")
    for e in rows2:
        md.append(f"| {e['label']} | {e['E_grid']:.2f} | {e['E_vertex']:.4f} | {e['E_ls']:.4f} | {e['Lam']:.3f} | {e.get('published','--')} | "
                  f"{e['nearest_ref']:.4f} | {pct(e['err'])} |")

    # ---------- convergence ----------
    conv = []
    ref = R["disk22_E1"]
    desc = {"conv_base_s0": "baseline P1: 3x128, N=5000, 1500 ep, mu=200, h=0.5 (seed 0)", "conv_base_s1": "baseline, seed 1",
            "conv_base_s2": "baseline, seed 2", "conv_P0_s0": "P0 (constant lr, early stop), seed 0",
            "conv_P0_s1": "P0, seed 1", "conv_P0_s2": "P0, seed 2", "conv_w32d2": "network 2x32", "conv_w64d2": "network 2x64",
            "conv_w256d4": "network 4x256", "conv_N2000": "N=2000", "conv_N20000": "N=20000", "conv_ep500": "500 epochs",
            "conv_ep5000": "5000 epochs", "conv_mu50": "mu=50", "conv_mu1000": "mu=1000", "conv_h025": "E-step h=0.25",
            "conv_h0125": "E-step h=0.125"}
    for name, dsc in desc.items():
        c = curve(name)
        if c is None or len(c[0]) < 3: continue
        E, L, r, d = c
        i = int(np.argmin(np.abs(E - 6.5)))
        ev, _, curv = vertex3(E, L, i)
        conv.append(dict(name=name, desc=dsc, E_vertex=ev, err_vertex=(ev - ref) / ref, Lam_65=float(L[i]),
                         core_frac=r[i]["core_frac"], E_ls=r[i]["E_ls"], err_ls=(r[i]["E_ls"] - ref) / ref,
                         RQ=r[i]["RQ"], err_RQ=(r[i]["RQ"] - ref) / ref, curv=curv,
                         E_ls_all=[x["E_ls"] for x in r], Lam_all=L.tolist(), E=E.tolist(),
                         wall_s=sum(x["wall_time"] for x in r), epochs=[x["epochs_run"] for x in r]))
    S["convergence"] = conv
    md.append("\n### Convergence study, disk p=2.2, E1 = %.5f (grid 6.0, 6.5, 7.0 unless stated)\n\n| setting | Lam(6.0) | Lam(6.5) | Lam(7.0) | core frac | 3-pt vertex | err %% | E_ls | err %% | RQ | err %% | wall min |\n|---|---|---|---|---|---|---|---|---|---|---|---|" % ref)
    for e in conv:
        Ls = dict(zip(e["E"], e["Lam_all"]))
        g = lambda x: f"{Ls[x]:.3f}" if x in Ls else "--"
        md.append(f"| {e['desc']} | {g(6.0)} | {e['Lam_65']:.3f} | {g(7.0)} | {e['core_frac']:.2f} | {e['E_vertex']:.4f} | "
                  f"{pct(e['err_vertex'])} | {e['E_ls']:.4f} | {pct(e['err_ls'])} | {e['RQ']:.4f} | {pct(e['err_RQ'])} | {e['wall_s']/60:.0f} |")

    # ---------- other p ----------
    op = []
    for name, p in [("disk_p1.5", 1.5), ("conv_base_s0", 2.2), ("disk_p3", 3.0), ("disk_p4", 4.0), ("disk_p4_w256d4", 4.0)]:
        e = est(name, refs[f"p{p}_d2"]["E1"])
        if e:
            e["p"] = p; op.append(e)
    S["other_p"] = op
    md.append("\n### Other p (unit disk, E1, baseline P1)\n\n| p | run | h | ref | 3-pt vertex | err % | E_ls | err % | RQ | err % | Lam_min | core frac |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in op:
        md.append(f"| {e['p']} | {e['name']} | {e['h']} | {e['ref']:.4f} | {e['E_vertex']:.4f} | {pct(e['err_E_vertex'])} | {e['E_ls']:.4f} | "
                  f"{pct(e['err_E_ls'])} | {e['RQ']:.4f} | {pct(e['err_RQ'])} | {e['Lam_min']:.3f} | {e['core_frac']:.2f} |")

    tot = 0.0
    for q in jobs.QUEUES.values():
        for j in q:
            d = load(j["name"])
            if d: tot += d.get("total_wall_s", 0)
    S["total_train_wall_h"] = tot / 3600
    md.append(f"\nTotal training wall time (sum over jobs, 1 thread each): {tot/3600:.1f} h")
    json.dump(S, open(RES / "plap_summary.json", "w"), indent=1)
    open(HERE / "tables_auto.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    figures(R, refs, REFS)


def figures(R, refs, S_REFS):
    # Fig 1: disk loss curves, P0 vs P1
    for dom, names, rr, fn, title in [
        ("disk", ["disk22_coarse_P0", "disk22_coarse_P1"], [a for a, _ in S_REFS["disk"]], "plap_disk_curves.png",
         "unit disk, p = 2.2"),
        ("square", ["square22_coarse_P0", "square22_coarse_P1"], [a for a, _ in S_REFS["square"]], "plap_square_curves.png",
         r"square $[0,2]^2$, p = 2.2"),
        ("ball", ["ball25_coarse_P1"], [R["ball_E1"], R["ball_E2"]], "plap_ball_curves.png", "unit ball in R^3, p = 2.5")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ok = False
        for n, st, lab in zip(names[::-1], ["o-", "s--"], ["lr decay (P1)", "constant lr + early stop (P0)"] if len(names) == 2 else ["lr decay (P1)"]):
            c = curve(n)
            if c is None: continue
            ok = True
            ax.plot(c[0], c[1], st, ms=4, lw=1.2, label=lab, color="C0" if "P1" in n else "C1")
            c2 = curve(n.replace("coarse_", "coarse2_"))
            if c2 is not None and "P1" in n:
                ax.plot(c2[0], c2[1], st, ms=4, lw=1.2, color="C0", alpha=0.6)
        for k, x in enumerate(rr):
            if x: ax.axvline(x, color="k", lw=0.8, ls=":", label="reference" if k == 0 else None)
        ax.set_yscale("log"); ax.set_xlabel("E"); ax.set_ylabel(r"$\bar\Lambda_p(E)$ (fresh sample)")
        ax.set_title(title); ax.legend(frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout()
        if ok: fig.savefig(FIG / fn, dpi=160)
        plt.close(fig)
    # Fig 2: local curves near E1 for the convergence configurations
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ref = R["disk22_E1"]
    for n, lab, col, st in [("conv_P0_s0", "P0 seed 0", "C1", "s--"), ("conv_P0_s1", "P0 seed 1", "C1", "^--"),
                            ("conv_P0_s2", "P0 seed 2", "C1", "v--"), ("conv_base_s0", "P1 seed 0", "C0", "o-"),
                            ("conv_base_s1", "P1 seed 1", "C0", "^-"), ("conv_base_s2", "P1 seed 2", "C0", "v-"),
                            ("conv_ep5000", "P1, 5000 epochs", "C2", "D-"), ("conv_N20000", "P1, N=20000", "C3", "P-"),
                            ("conv_w256d4", "P1, 4x256", "C4", "X-")]:
        c = curve(n)
        if c is None: continue
        axs[0].plot(c[0], c[1], st, color=col, ms=5, lw=1.1, label=lab)
        axs[1].plot(c[0], [x["E_ls"] for x in c[2]], st, color=col, ms=5, lw=1.1)
    xx = np.linspace(5.5, 7.5, 100)
    axs[0].axvline(ref, color="k", ls=":", lw=0.8); axs[1].axhline(ref, color="k", ls=":", lw=0.8, label="reference")
    axs[0].set_xlabel("E"); axs[0].set_ylabel(r"$\bar\Lambda_p(E)$ (fresh sample)"); axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_title("loss curve near $E_1$ (disk, p=2.2)")
    axs[1].set_xlabel("E (training value)"); axs[1].set_ylabel(r"$E_{\rm ls}$ of the trained network")
    axs[1].set_title("frozen-network least-squares estimate"); axs[1].legend(frameon=False)
    for a in axs: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "plap_convergence_curves.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
