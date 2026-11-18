"""Run FEM eigenvalue computations for all cases; results -> experiments/results/fem_<case>.json
usage: python -m experiments.fem.run_fem CASE [CASE...]   (1 thread; resumable per level)"""
import sys, json, time, os
import numpy as np
from pathlib import Path
from skfem import MeshTri2, MeshTet2
from scipy.special import jn_zeros
from experiments.fem import fem_lib as F

RES = Path(__file__).resolve().parent.parent / "results"

def annulus2(nref, r_in=0.5, r_out=1.0):
    """Quadratic (isoparametric) mesh of the annulus: boundary edge midpoints projected onto the circles."""
    m1 = F.mesh_annulus(nref, r_in, r_out)
    m = MeshTri2.from_mesh(m1)
    p = m.p.copy()
    nv = m1.p.shape[1]
    for f in m1.boundary_facets():
        a, b = m1.facets[:, f]
        mid = 0.5 * (m1.p[:, a] + m1.p[:, b])
        R = r_in if abs(np.hypot(*m1.p[:, a]) - r_in) < 1e-9 else r_out
        p[:, nv + f] = mid / np.hypot(*mid) * R
    return MeshTri2(p, m.t)

# case: name -> dict(mesh=builder, V=potential, levels, k, degree)
CASES = {
 "disk_free":   dict(mesh=lambda n: MeshTri2.init_circle(n), V=F.V_zero, levels=[3, 4, 5, 6], k=12),
 "disk_free_P1":dict(mesh=lambda n: F.mesh_disk(n), V=F.V_zero, levels=[4, 5, 6, 7], k=12, degree=1),
 "ball_free":   dict(mesh=lambda n: MeshTet2.init_ball(n), V=F.V_zero, levels=[2, 3, 4], k=60),
 "double_well": dict(mesh=lambda n: MeshTet2.init_ball(n), V=F.V_double_well(), levels=[2, 3, 4], k=60),
 "four_well":   dict(mesh=lambda n: MeshTet2.init_ball(n), V=F.V_four_well(), levels=[2, 3, 4], k=60),
 "harm_w1":     dict(mesh=lambda n: MeshTri2.init_circle(n), V=F.V_harmonic(1.0), levels=[3, 4, 5, 6], k=30),
 "harm_w10":    dict(mesh=lambda n: MeshTri2.init_circle(n), V=F.V_harmonic(10.0), levels=[3, 4, 5, 6], k=30),
 "harm_w1_orig": dict(mesh=lambda n: MeshTri2.init_circle(n), V=F.V_harmonic(1.0, 1.0), levels=[4, 5, 6], k=12),
 "harm_w10_orig": dict(mesh=lambda n: MeshTri2.init_circle(n), V=F.V_harmonic(10.0, 1.0), levels=[4, 5, 6], k=12),
 "annulus":     dict(mesh=annulus2, V=F.V_zero, levels=[2, 3, 4, 5], k=20),
 "square":      dict(mesh=lambda n: F.mesh_square(n), V=F.V_zero, levels=[3, 4, 5, 6], k=20),
 "triangle":    dict(mesh=lambda n: F.mesh_triangle(n), V=F.V_zero, levels=[3, 4, 5, 6], k=20),
}

def run(case):
    c = CASES[case]
    path = RES / f"fem_{case}.json"
    out = json.load(open(path)) if path.exists() else dict(case=case, degree=c.get("degree", 2), levels={})
    for lev in c["levels"]:
        if str(lev) in out["levels"]:
            continue
        t0 = time.perf_counter()
        m = c["mesh"](lev)
        vals, info = F.solve(m, c["V"], k=c["k"], degree=c.get("degree", 2),
                             sigma=c.get("sigma", -10.0 if "square" in case or "triangle" in case else 0.0 if "well" in case else -5.0))
        info["wall"] = time.perf_counter() - t0
        out["levels"][str(lev)] = dict(eigenvalues=vals.tolist(), **info)
        json.dump(out, open(path, "w"), indent=1)
        print(case, lev, info["ndof"], f"{info['wall']:.1f}s", np.round(vals[:6], 5), flush=True)

if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
