"""Post-process FEM results: Richardson extrapolation, exact references, multiplicities.
Writes experiments/results/fem_summary.json"""
import json
import numpy as np
from pathlib import Path
from scipy.special import jn_zeros, jv, yv, spherical_jn
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

RES = Path(__file__).resolve().parent.parent / "results"

def load(case):
    d = json.load(open(RES / f"fem_{case}.json"))
    lv = sorted(int(k) for k in d["levels"])
    return lv, [np.array(d["levels"][str(l)]["eigenvalues"]) for l in lv], [d["levels"][str(l)] for l in lv]

def extrapolate(vals, ratio=2.0, pmin=1.5, pmax=4.5, p_default=None):
    """vals: list of eigenvalue arrays on successive levels (coarse->fine). Uses last three.
    Returns E*, err, p_obs arrays."""
    a, b, c = vals[-3], vals[-2], vals[-1]
    n = min(len(a), len(b), len(c))
    a, b, c = a[:n], b[:n], c[:n]
    d1, d2 = a - b, b - c
    with np.errstate(all="ignore"):
        p = np.log(np.abs(d1) / np.abs(d2)) / np.log(ratio)
    ok = np.isfinite(p) & (d1 * d2 > 0)
    pe = np.where(ok, np.clip(p, pmin, pmax), p_default if p_default else 2.0)
    Es = c - d2 / (ratio ** pe - 1.0)          # c + (c-b)/(r^p-1)  (d2 = b-c)
    Es = c + (c - b) / (ratio ** pe - 1.0)
    err = np.maximum(np.abs(Es - c), 1e-12)
    return Es, err, p

def multiplicities(Es, vals_prev, vals_last, tol_rel=0.1, tol_abs=1e-4):
    """Group sorted eigenvalues into clusters of numerically-degenerate ones using the finest-level values
    (mesh asymmetry splits true multiplets by an amount ~ discretisation error)."""
    n = len(Es)
    delta = np.abs(vals_prev[:n] - vals_last[:n])
    groups = [[0]]
    for i in range(1, n):
        j = groups[-1][-1]
        tol = tol_rel * max(delta[i], delta[j]) + tol_abs
        if vals_last[i] - vals_last[j] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups

# ------------------------------------------------------------------ exact references
def exact_disk(n, r=1.0):
    ev = []
    for m in range(0, 12):
        for k in jn_zeros(m, 6):
            ev += [(k * k, 1 if m == 0 else 2, f"m={m}")]
    ev.sort(key=lambda t: t[0]); return ev

def exact_ball(nmax=12):
    ev = []
    for l in range(0, 10):
        xs = np.linspace(0.1, 40, 4000); f = spherical_jn(l, xs)
        for i in range(len(xs) - 1):
            if f[i] * f[i + 1] < 0:
                z = brentq(lambda x: spherical_jn(l, x), xs[i], xs[i + 1], xtol=1e-14)
                ev.append((z * z, 2 * l + 1, f"l={l}"))
    ev.sort(key=lambda t: t[0]); return ev

def exact_annulus(a=0.5, b=1.0):
    ev = []
    for m in range(0, 10):
        f = lambda k: jv(m, k * a) * yv(m, k * b) - jv(m, k * b) * yv(m, k * a)
        ks = np.linspace(0.5, 12, 4000); fv = f(ks)
        for i in range(len(ks) - 1):
            if fv[i] * fv[i + 1] < 0:
                z = brentq(f, ks[i], ks[i + 1], xtol=1e-14)
                ev.append((z * z, 1 if m == 0 else 2, f"m={m}"))
    ev.sort(key=lambda t: t[0]); return ev

def exact_square():
    d = {}
    for m in range(1, 15):
        for n in range(1, 15):
            d.setdefault(m * m + n * n, []).append((m, n))
    return [(np.pi ** 2 * s, len(v), str(v)) for s, v in sorted(d.items())]

def exact_triangle():
    d = {}
    for m in range(2, 15):
        for n in range(1, m):
            d.setdefault(m * m + n * n, []).append((m, n))
    return [(np.pi ** 2 * s, len(v), str(v)) for s, v in sorted(d.items())]

def radial_harmonic(omega, coef=0.5, N=(2000, 4000), mmax=8, nk=4):
    """Independent reference (as audit/scripts/potentials_reference.py): finite-volume radial ODE per angular
    momentum m, Richardson-extrapolated. Returns sorted (E, mult, label)."""
    def radial(m, N):
        h = 1.0 / N; r = (np.arange(N) + 0.5) * h; rp = r + h / 2; rm = r - h / 2
        d = (rp + rm) / (r * h * h) + m * m / r ** 2 + coef * omega ** 2 * r ** 2
        d[-1] += rp[-1] / (r[-1] * h * h)
        e = -rp[:-1] / (np.sqrt(r[:-1] * r[1:]) * h * h)
        return eigh_tridiagonal(d, e, select="i", select_range=(0, nk - 1))[0]
    ev = []
    for m in range(mmax):
        a, b = radial(m, N[0]), radial(m, N[1]); x = (4 * b - a) / 3
        ev += [(x[k], 1 if m == 0 else 2, f"m={m},k={k+1}") for k in range(nk)]
    ev.sort(key=lambda t: t[0]); return ev

def match(groups_E, exact):
    """pair each computed group with exact cluster by order (both sorted)."""
    return exact

def summarize(case, emax=None, ngroups=None):
    lv, vals, infos = load(case)
    Es, err, p = extrapolate(vals)
    n = len(Es)
    groups = multiplicities(Es, vals[-2], vals[-1])
    rows = []
    for g in groups:
        if emax is not None and Es[g[0]] > emax: break
        rows.append(dict(E=float(np.mean(Es[g])), err=float(np.max(err[g])), mult=len(g),
                         spread_fine=float(vals[-1][g].max() - vals[-1][g].min()),
                         E_fine=float(np.mean(vals[-1][g])), E_prev=float(np.mean(vals[-2][g])),
                         p_obs=float(np.nanmedian(p[g]))))
    return dict(case=case, levels=lv, ndof=[i["ndof"] for i in infos],
                wall=[i["wall"] for i in infos], rows=rows,
                all_extrap=Es.tolist(), all_err=err.tolist(), all_p=[None if not np.isfinite(x) else float(x) for x in p],
                last_group_complete=bool(len(groups[-1]) < n))

def main():
    out = {}
    refs = dict(disk_free=exact_disk(None), ball_free=exact_ball(), annulus=exact_annulus(),
                square=exact_square(), triangle=exact_triangle(),
                harm_w1=radial_harmonic(1.0), harm_w10=radial_harmonic(10.0),
                harm_w1_orig=radial_harmonic(1.0, 1.0), harm_w10_orig=radial_harmonic(10.0, 1.0))
    emax = dict(disk_free=60, ball_free=75, annulus=60, square=120, triangle=120, harm_w1=55, harm_w10=70,
                harm_w1_orig=55, harm_w10_orig=70, double_well=75, four_well=75)
    for case in ["disk_free", "ball_free", "annulus", "square", "triangle", "harm_w1", "harm_w10",
                 "harm_w1_orig", "harm_w10_orig", "double_well", "four_well"]:
        s = summarize(case, emax=emax[case])
        if case in refs:
            ex = refs[case]
            for r_, e_ in zip(s["rows"], ex):
                r_["exact"] = float(e_[0]); r_["exact_mult"] = e_[1]; r_["label"] = e_[2]
        out[case] = s
    json.dump(out, open(RES / "fem_summary.json", "w"), indent=1)
    for c, s in out.items():
        print("==", c, s["levels"], s["ndof"])
        for r in s["rows"][:14]:
            ex = f" exact={r['exact']:.6f}({r['exact_mult']}) {r['label']} errFEM={r['E']-r['exact']:+.2e}" if "exact" in r else ""
            print(f"  {r['E']:.5f} +-{r['err']:.1e} x{r['mult']} p={r['p_obs']:.2f}{ex}")

if __name__ == "__main__":
    main()
