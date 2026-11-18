"""Reference for the REGULARISED radial problem on the unit ball of R^d:

    (r^{d-1} phi(u'))' + E r^{d-1} |u|^{p-2}u = 0,   phi(s) = (s^2+eps^2)^{(p-2)/2} s,   u'(0)=0, u(1)=0, u>0,
    normalisation  |S^{d-1}| int_0^1 |u|^p r^{d-1} dr = 1.

The regularised operator is NOT homogeneous, so the "eigenvalue" depends on the amplitude a=u(0), i.e. on the normalisation:
for each a we find E(a) such that the first zero of u is r=1 (brentq), then a such that the mass is 1 (brentq).
Also tabulated: dE/dmass sensitivity (E at mass 0.98 and 1.02).   Run: python -m experiments.plap.refs_reg
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def phi_inv(y, p, eps):
    s = -1.0 if y < 0 else 1.0; y = abs(y)
    if y == 0.0: return 0.0
    f = lambda t: (t * t + eps * eps) ** ((p - 2) / 2) * t - y
    hi = max(y ** (1 / (p - 1)), 1e-300)
    while f(hi) < 0: hi *= 2
    return s * brentq(f, 0.0, hi, xtol=1e-15, rtol=1e-14)


def solve(E, a, p, d, eps, tol=1e-10, r0=1e-6):
    def f(r, y):
        u, w, m = y
        up = abs(u) ** (p - 1) * np.sign(u)
        return [phi_inv(w / r ** (d - 1), p, eps), -E * r ** (d - 1) * up, abs(u) ** p * r ** (d - 1)]
    y0 = [a, -E * a ** (p - 1) * r0 ** d / d, a ** p * r0 ** d / d]
    return solve_ivp(f, [r0, 1.0], y0, rtol=tol, atol=tol * 1e-3, method="DOP853")


def E_of_a(a, p, d, eps, lo, hi):
    g = lambda E: solve(E, a, p, d, eps).y[0, -1]
    # first eigen-branch: u(1) > 0 for small E, < 0 just above
    return brentq(g, lo, hi, xtol=1e-9)


def reference(p, d, eps, E0, mass=1.0):
    area = {2: 2 * np.pi, 3: 4 * np.pi}[d]
    def mass_err(a):
        E = E_of_a(a, p, d, eps, 0.3 * E0, 2.0 * E0)
        return area * solve(E, a, p, d, eps).y[2, -1] - mass
    a = brentq(mass_err, 0.05, 5.0, xtol=1e-8)
    return E_of_a(a, p, d, eps, 0.3 * E0, 2.0 * E0), a


def main():
    refs = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "plap_refs.json")))
    out = {}
    for p, epss in [(1.5, [0.3, 0.1, 0.03, 0.01]), (4.0, [0.3, 0.1])]:
        E0 = refs[f"p{p}_d2"]["E1"]
        for eps in epss:
            E, a = reference(p, 2, eps, E0)
            Em, _ = reference(p, 2, eps, E0, 0.98); Ep, _ = reference(p, 2, eps, E0, 1.02)
            out[f"p{p}_eps{eps}"] = dict(p=p, eps=eps, E_eps=E, u0=a, E1=E0, E_mass098=Em, E_mass102=Ep)
            print(f"p={p} eps={eps:5.2f}: E_eps={E:.5f} (u(0)={a:.4f}); E1={E0:.5f}; shift {100*(E-E0)/E0:+.2f}%; "
                  f"mass 0.98/1.02 -> {Em:.5f}/{Ep:.5f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "plap_refs_reg.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
