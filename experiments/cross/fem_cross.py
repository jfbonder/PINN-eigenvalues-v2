"""FEM (scikit-fem P2) for Dirichlet Laplacian on the cross / L-shape with graded refinement at re-entrant corners.
usage: python -m experiments.cross.fem_cross  -> experiments/results/cross_fem.json"""
import json, sys, time
import numpy as np
import scipy.sparse.linalg as sla
from pathlib import Path
from skfem import MeshTri, Basis, ElementTriP2, BilinearForm, LinearForm
from skfem.helpers import dot, grad

RES = Path(__file__).resolve().parent.parent / "results"
A = 0.25

def inside_cross(x, y, a=A):
    return ((abs(x) <= 1 + 1e-12) & (abs(y) <= a + 1e-12)) | ((abs(x) <= a + 1e-12) & (abs(y) <= 1 + 1e-12))
def inside_L(x, y):
    return (abs(x) <= 1 + 1e-12) & (abs(y) <= 1 + 1e-12) & ~((x > 1e-12) & (y > 1e-12))

def base_mesh(kind):
    if kind == "cross":
        xs = np.array([-1, -0.625, -A, 0.0, A, 0.625, 1.0]); inside = inside_cross
        corners = [(sx * A, sy * A) for sx in (-1, 1) for sy in (-1, 1)]
    else:
        xs = np.array([-1, -0.5, 0.0, 0.5, 1.0]); inside = inside_L; corners = [(0.0, 0.0)]
    X, Y = np.meshgrid(xs, xs, indexing="ij"); n = len(xs)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (xs[j] + xs[j + 1])
            if inside(cx, cy):
                a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
                tris += [[a, b, c], [a, c, d]]
    t = np.array(tris).T; used = np.unique(t); remap = -np.ones(n * n, int); remap[used] = np.arange(len(used))
    p = np.vstack([X.ravel()[used], Y.ravel()[used]])
    return MeshTri(p, remap[t]), corners

def graded(kind, nuni, ngrade):
    m, corners = base_mesh(kind)
    m = m.refined(nuni)
    for k in range(ngrade):           # refine elements whose centroid is within R_k of a corner
        cen = m.p[:, m.t].mean(1)
        R = 0.5 * 0.5 ** (k * 0.5)
        d = np.min([np.hypot(cen[0] - cx, cen[1] - cy) for cx, cy in corners], axis=0)
        m = m.refined(np.where(d < R)[0])
    return m

def solve(m, k=12, sigma=0.0):
    b = Basis(m, ElementTriP2())
    K = BilinearForm(lambda u, v, w: dot(grad(u), grad(v))).assemble(b)
    M = BilinearForm(lambda u, v, w: u * v).assemble(b)
    I = b.complement_dofs(b.get_dofs())
    v, vec = sla.eigsh(K[I][:, I].tocsc(), k=k, M=M[I][:, I].tocsc(), sigma=sigma, which="LM")
    o = np.argsort(v); return v[o], b, I, vec[:, o]

def run(kind):
    out = dict(kind=kind, levels=[])
    for nuni, ng in [(1, 4), (2, 6), (3, 8), (3, 11), (4, 13)]:
        t0 = time.perf_counter(); m = graded(kind, nuni, ng)
        v, *_ = solve(m, 12, sigma=0.5)
        out["levels"].append(dict(nuni=nuni, ngrade=ng, nelem=int(m.t.shape[1]), eig=v.tolist(), wall=time.perf_counter() - t0))
        print(kind, nuni, ng, m.t.shape[1], np.round(v[:8], 5), f"{time.perf_counter()-t0:.1f}s", flush=True)
    json.dump(out, open(RES / f"cross_fem_{kind}.json", "w"), indent=1)

if __name__ == "__main__":
    for k in sys.argv[1:] or ["cross", "L"]: run(k)
