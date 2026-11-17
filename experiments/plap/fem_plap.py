"""Independent P1 finite-element reference for FIRST eigenvalues of the p-Laplacian on convex 2-D domains
(and axisymmetric 3-D domains), by direct minimisation of the Rayleigh quotient

        R(u) = int |grad u|^p w / int |u|^p w ,      u in P1, u = 0 on the Dirichlet part,   w = 1 (2-D) or rho (axisym.)

with L-BFGS (float64, exact gradients by autograd), nested meshes (coarse solution interpolated as initial guess)
and Richardson extrapolation in h.  The first eigenfunction is the unique positive critical point, and for u>0 the
quotient has no other critical points, so descent from a positive start gives lambda_1 of the discrete problem,
which is an upper bound of lambda_1(Omega_h) >= lambda_1(Omega) up to the quadrature of |u|^p (degree-4 rule).

Used for:  lambda_1(disk) [validation against shooting], lambda_1(half disk) (= upper bound of lambda_2(disk);
equal to it if the 2nd eigenfunction is antisymmetric, as found by Horak), lambda_1(square [0,2]^2),
lambda_1([0,1]x[0,2]) and lambda_1(triangle (0,0),(2,0),(2,2)) (upper bounds for lambda_2([0,2]^2)),
3-D unit ball (axisymmetric; validation) and 3-D half ball (upper bound for lambda_2(ball)).

Run: python -m experiments.plap.fem_plap [--nmax 160]   -> experiments/results/plap_fem_refs.json
"""
import argparse, json, os, time
import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

torch.set_default_dtype(torch.float64)
A_, B_ = 0.445948490915965, 0.091576213509771
QP = np.array([[A_, A_, 1 - 2 * A_], [A_, 1 - 2 * A_, A_], [1 - 2 * A_, A_, A_],
               [B_, B_, 1 - 2 * B_], [B_, 1 - 2 * B_, B_], [1 - 2 * B_, B_, B_]])
QW = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def sector_points(n, th0, th1):
    """Ring points of the unit-disk sector th0<=theta<=th1 (full disk if th1-th0=2pi)."""
    pts = [[0.0, 0.0]]
    full = abs(th1 - th0 - 2 * np.pi) < 1e-12
    for i in range(1, n + 1):
        r = i / n
        m = max(1, int(round(i * (th1 - th0) / (np.pi / 3) )))
        th = np.linspace(th0, th1, m, endpoint=False) if full else np.linspace(th0, th1, m + 1)
        pts += [[r * np.cos(t), r * np.sin(t)] for t in th]
    return np.array(pts)


def grid_points(n, kind):
    if kind == "square2":
        x = np.linspace(0, 2, 2 * n + 1); X, Y = np.meshgrid(x, x)
    elif kind == "rect":
        x = np.linspace(0, 1, n + 1); y = np.linspace(0, 2, 2 * n + 1); X, Y = np.meshgrid(x, y)
    elif kind == "tri":
        x = np.linspace(0, 2, 2 * n + 1); X, Y = np.meshgrid(x, x); k = Y <= X + 1e-12; X, Y = X[k], Y[k]
    return np.c_[X.ravel(), Y.ravel()]


def structured_tris(pts, kind, n):
    """Delaunay is ambiguous on structured grids but any choice is a valid triangulation of a convex set."""
    return Delaunay(pts).simplices


DOM = {
    # name: (points(n), dirichlet(x,y)->bool, weight)
    "disk":      (lambda n: sector_points(n, 0, 2 * np.pi), lambda x, y: x * x + y * y > 1 - 1e-9, None),
    "halfdisk":  (lambda n: sector_points(n, 0, np.pi), lambda x, y: (x * x + y * y > 1 - 1e-9) | (y < 1e-12), None),
    "square2":   (lambda n: grid_points(n, "square2"),
                  lambda x, y: (x < 1e-12) | (x > 2 - 1e-12) | (y < 1e-12) | (y > 2 - 1e-12), None),
    "rect1x2":   (lambda n: grid_points(n, "rect"),
                  lambda x, y: (x < 1e-12) | (x > 1 - 1e-12) | (y < 1e-12) | (y > 2 - 1e-12), None),
    "tri_diag":  (lambda n: grid_points(n, "tri"),
                  lambda x, y: (y < 1e-12) | (x > 2 - 1e-12) | (np.abs(x - y) < 1e-12), None),
    # axisymmetric: coordinates (rho, z) = (x, y), weight rho, natural condition on rho = 0
    "ball3_axi":     (lambda n: sector_points(n, -np.pi / 2, np.pi / 2), lambda x, y: x * x + y * y > 1 - 1e-9, "rho"),
    "halfball3_axi": (lambda n: sector_points(n, 0, np.pi / 2),
                      lambda x, y: (x * x + y * y > 1 - 1e-9) | (y < 1e-12), "rho"),
}


def assemble(pts, tri, weight):
    P = pts[tri]                                    # [T,3,2]
    d1 = P[:, 1] - P[:, 0]; d2 = P[:, 2] - P[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    keep = np.abs(det) > 1e-14
    tri, P, d1, d2, det = tri[keep], P[keep], d1[keep], d2[keep], det[keep]
    area = 0.5 * np.abs(det)
    # gradients of barycentric functions
    g1 = np.c_[d2[:, 1], -d2[:, 0]] / det[:, None]; g2 = np.c_[-d1[:, 1], d1[:, 0]] / det[:, None]
    g0 = -g1 - g2
    G = np.stack([g0, g1, g2], 1)                   # [T,3,2]
    xq = np.einsum("qk,tkc->tqc", QP, P)            # quad points [T,6,2]
    wq = np.ones(xq.shape[:2]) if weight is None else xq[:, :, 0]
    wbar = wq @ QW                                  # exact int of the (linear) weight / area
    t = torch.as_tensor
    return dict(tri=t(tri), G=t(G), area=t(area), wq=t(wq * QW[None, :]), wbar=t(wbar), QP=t(QP))


def rayleigh(u, M, p):
    ut = u[M["tri"]]                                # [T,3]
    grad = torch.einsum("tk,tkc->tc", ut, M["G"])
    num = (M["area"] * M["wbar"] * (grad.pow(2).sum(1) + 1e-300).pow(p / 2)).sum()
    uq = ut @ M["QP"].T                             # [T,6]
    den = (M["area"][:, None] * M["wq"] * (uq.abs() + 1e-300).pow(p)).sum()
    return num / den


def solve(name, p, n, u0=None, pts0=None, iters=60, verbose=False):
    ptsf, dirf, weight = DOM[name]
    pts = ptsf(n)
    tri = Delaunay(pts).simplices
    M = assemble(pts, tri, weight)
    dirichlet = dirf(pts[:, 0], pts[:, 1])
    free = torch.as_tensor(~dirichlet)
    if u0 is None:
        # positive start: product of distances-like bump
        r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
        start = np.ones(len(pts))
    else:
        start = LinearNDInterpolator(pts0, u0, fill_value=0.0)(pts)
    start = np.where(dirichlet, 0.0, np.maximum(start, 1e-3))
    v = torch.tensor(start[~dirichlet], requires_grad=True)

    def full(v):
        u = torch.zeros(len(pts)); u = u.masked_scatter(free, v); return u

    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=500, history_size=50, tolerance_grad=1e-12,
                            tolerance_change=1e-15, line_search_fn="strong_wolfe")
    last = None
    for it in range(iters):
        def closure():
            opt.zero_grad(); L = rayleigh(full(v), M, p); L.backward(); return L
        L = opt.step(closure).item()
        with torch.no_grad():
            v /= v.abs().max()
        if last is not None and abs(last - L) < 1e-11 * abs(L):
            break
        last = L
    lam = rayleigh(full(v), M, p).item()
    return lam, pts, full(v).detach().numpy(), len(pts)


def richardson(hs, lams):
    """lam(h) = lam0 + C h^q from the three finest meshes (h halved each time)."""
    l1, l2, l3 = lams[-3:]
    ratio = (l1 - l2) / (l2 - l3)
    q = np.log2(ratio) if ratio > 1 else float("nan")
    lam0 = l3 - (l2 - l3) / (ratio - 1) if ratio > 1 else float("nan")
    return float(lam0), float(q)


def run(name, p, ns):
    u0 = pts0 = None; lams = []; rows = []
    for n in ns:
        t0 = time.time()
        lam, pts0, u0, nn = solve(name, p, n, u0, pts0)
        lams.append(lam); rows.append(dict(n=n, nodes=nn, lam=lam, t=time.time() - t0))
        print(f"  {name:14s} p={p} n={n:4d} nodes={nn:7d} lam_h={lam:.6f} ({time.time()-t0:.0f}s)", flush=True)
    lam0, q = richardson(ns, lams)
    print(f"  -> extrapolated {lam0:.5f} (observed order {q:.2f})", flush=True)
    return dict(domain=name, p=p, meshes=rows, extrap=lam0, order=q, finest=lams[-1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nmax", type=int, default=160)
    ap.add_argument("--only", type=str, default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    ns = [n for n in (20, 40, 80, 160, 320) if n <= a.nmax]
    cases = [("disk", 2.0), ("disk", 2.2), ("halfdisk", 2.0), ("halfdisk", 2.2), ("square2", 2.2), ("rect1x2", 2.2),
             ("tri_diag", 2.2), ("ball3_axi", 2.5), ("halfball3_axi", 2.0), ("halfball3_axi", 2.5),
             ("disk", 1.5), ("disk", 3.0), ("disk", 4.0)]
    path = os.path.join(os.path.dirname(__file__), "..", "results", "plap_fem_refs.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name, p in cases:
        key = f"{name}_p{p}"
        if a.only and a.only not in key:
            continue
        if key in out and out[key]["meshes"][-1]["n"] >= ns[-1]:
            continue
        out[key] = run(name, p, ns)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
