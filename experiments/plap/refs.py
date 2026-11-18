"""Reference radial eigenvalues of the p-Laplacian on the unit ball of R^d by shooting.

ODE (r^{d-1}|u'|^{p-2}u')' + E r^{d-1}|u|^{p-2}u = 0, u(0)=1, u'(0)=0; E_1 = value of E for which the first
zero of u is r=1.  By homogeneity/scaling u_E(r)=u_1(E^{1/p} r), so E_k = z_k^p with z_k the k-th zero of the
E=1 solution: one IVP solve gives every radial eigenvalue (no root bracketing needed).
Start-up at r0 with a 2-term series: u = 1 - c r^q + c2 r^{2q}, q=p/(p-1), c=(q^-1)(1/d)^{1/(p-1)}.
Checks: p=2 against Bessel zeros (j_{0,1}^2, pi^2), and independence of (r0, tol).

Run: python -m experiments.plap.refs   -> experiments/results/plap_refs.json
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp


def zeros_unit(p, d, r0=1e-6, tol=1e-12, rmax=12.0, nz=2):
    q = p / (p - 1.0)
    c = (1.0 / d) ** (1.0 / (p - 1.0)) / q

    def f(r, y):
        u, w = y            # w = r^{d-1}|u'|^{p-2}u'
        return [np.sign(w) * abs(w / r ** (d - 1)) ** (1.0 / (p - 1.0)),
                -r ** (d - 1) * np.sign(u) * abs(u) ** (p - 1.0)]

    # first-order start-up: u ~ 1 - c r^q ; w = -int_0^r s^{d-1} u^{p-1} ds ~ -(r^d/d)(1 - (p-1)c d r^q/(d+q))
    u0 = 1.0 - c * r0 ** q
    w0 = -(r0 ** d / d) * (1.0 - (p - 1.0) * c * d * r0 ** q / (d + q))
    ev = lambda r, y: y[0]
    sol = solve_ivp(f, [r0, rmax], [u0, w0], rtol=tol, atol=tol * 1e-3, method="DOP853", events=ev)
    return sol.t_events[0][:nz]


def radial_eigs(p, d, **kw):
    return [float(z ** p) for z in zeros_unit(p, d, **kw)]


def main():
    out = {}
    print(f"{'p':>5} {'d':>2} {'E1 (radial, = lambda_1)':>26} {'spread':>9} {'2nd radial':>12}")
    for d in (2, 3):
        for p in (1.5, 2.0, 2.2, 2.5, 3.0, 4.0):
            vals = [radial_eigs(p, d, r0=r0, tol=tol) for r0, tol in [(1e-4, 1e-10), (1e-6, 1e-12), (1e-8, 1e-13)]]
            e1 = vals[1][0]; spread = max(abs(v[0] - e1) for v in vals)
            out[f"p{p}_d{d}"] = dict(p=p, d=d, E1=e1, E_rad2=vals[1][1], spread=spread)
            print(f"{p:5.1f} {d:2d} {e1:26.10f} {spread:9.1e} {vals[1][1]:12.6f}")
    from scipy.special import jn_zeros
    out["check_p2"] = dict(d2_exact=float(jn_zeros(0, 1)[0] ** 2), d3_exact=float(np.pi ** 2),
                           d2_rad2_exact=float(jn_zeros(0, 2)[1] ** 2), d3_rad2_exact=float(4 * np.pi ** 2))
    print("p=2 exact:", out["check_p2"])
    out["literature"] = dict(disk_p2p2_E1_Horak=6.5320, disk_p2p2_E2_Horak=18.395,
                             square02_p2p2_E1_Horak=5.4952, square02_p2p2_E2_Horak=15.144)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "plap_refs.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", os.path.abspath(path))


if __name__ == "__main__":
    main()
