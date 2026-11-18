"""Finite-element eigenvalue solver (scikit-fem, Lagrange P1/P2, shift-invert eigsh).

Problem:  -Delta u + V u = E u in Omega,  u = 0 on dOmega   (Dirichlet).
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse.linalg as sla
from skfem import (MeshTri, MeshTet, Basis, BilinearForm, ElementTriP1, ElementTriP2,
                   ElementTetP1, ElementTetP2, condense)
from skfem.helpers import dot, grad


# ---------------------------------------------------------------- potentials (numpy, x: (d,...))
def V_zero(x):
    return np.zeros(x.shape[1:])

def V_harmonic(omega, coef=0.5):
    """coef=0.5 -> (omega^2/2)|x|^2 (paper text & pinn_eigen.py); coef=1 -> omega^2|x|^2 (original notebook code)."""
    return lambda x: coef * omega ** 2 * (x ** 2).sum(0)

def V_double_well(a=1.0, sigma=0.5, V0=1.0):
    def V(x):
        r1 = (x[0] - a) ** 2 + x[1] ** 2 + x[2] ** 2
        r2 = (x[0] + a) ** 2 + x[1] ** 2 + x[2] ** 2
        return -V0 * (np.exp(-r1 / (2 * sigma ** 2)) + np.exp(-r2 / (2 * sigma ** 2)))
    return V

def V_four_well(a=0.22, sigma=0.2, V0=1.0):
    def V(x):
        out = np.zeros(x.shape[1:])
        for cx, cy in [(a, a), (-a, a), (a, -a), (-a, -a)]:
            rr = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2 + x[2] ** 2) / sigma
            m = rr < 1
            out[m] -= V0 * np.exp(-1.0 / (1.0 - rr[m] ** 2))
        return out
    return V


# ---------------------------------------------------------------- meshes
def mesh_disk(nref):
    return MeshTri.init_circle(nref)

def mesh_annulus(nref, r_in=0.5, r_out=1.0):
    """Structured polar mesh, triangles; radial layers geometric-free (uniform), n_theta ~ 2 pi r/h."""
    nr = 2 ** nref * 2          # radial intervals
    nt = int(round(2 * np.pi * (r_in + r_out) / 2 / ((r_out - r_in) / nr)))
    r = np.linspace(r_in, r_out, nr + 1)
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    R, T = np.meshgrid(r, th, indexing="ij")
    p = np.vstack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()])
    idx = lambda i, j: i * nt + (j % nt)
    tris = []
    for i in range(nr):
        for j in range(nt):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    return MeshTri(p, np.array(tris).T)

def mesh_square(nref):
    return MeshTri.init_sqsymmetric().refined(nref)

def mesh_triangle(nref):
    return MeshTri(np.array([[0., 1., 0.], [0., 0., 1.]]), np.array([[0], [1], [2]])).refined(nref)

def mesh_ball(nref):
    return MeshTet.init_ball(nref)

ELEMS = {("tri", 1): ElementTriP1, ("tri", 2): ElementTriP2,
         ("tet", 1): ElementTetP1, ("tet", 2): ElementTetP2}


# ---------------------------------------------------------------- solver
def solve(mesh, V, k=30, degree=2, sigma=None, intorder=None, return_vecs=False):
    """Return (eigenvalues sorted, info). sigma: shift for shift-invert (default: below spectrum)."""
    kind = "tet" if mesh.dim() == 3 else "tri"
    t0 = time.perf_counter()
    basis = Basis(mesh, ELEMS[(kind, degree)](), intorder=intorder or 2 * degree + 2)

    @BilinearForm
    def a(u, v, w):
        return dot(grad(u), grad(v)) + V(w.x) * u * v

    @BilinearForm
    def m(u, v, w):
        return u * v

    K = a.assemble(basis)
    M = m.assemble(basis)
    D = basis.get_dofs()
    I = basis.complement_dofs(D)
    Kc, Mc = K[I][:, I].tocsc(), M[I][:, I].tocsc()
    t1 = time.perf_counter()
    if sigma is None:
        sigma = -50.0 if V(np.zeros((mesh.dim(), 1))).min() >= -1e-9 and True else -50.0
    vals, vecs = sla.eigsh(Kc, k=k, M=Mc, sigma=sigma, which="LM")
    o = np.argsort(vals)
    t2 = time.perf_counter()
    info = dict(ndof=int(Kc.shape[0]), nelem=int(mesh.t.shape[1]), degree=degree,
                t_assemble=t1 - t0, t_solve=t2 - t1)
    if return_vecs:
        return vals[o], info, (basis, I, vecs[:, o])
    return vals[o], info


# ---------------------------------------------------------------- clustering / multiplicity
def cluster(vals, rel=2e-3, abs_=2e-3):
    """Group computed eigenvalues into clusters of numerically-equal values.
    Returns list of (mean value, multiplicity)."""
    out = []
    for v in vals:
        if out and abs(v - out[-1][0] / out[-1][1]) <= max(abs_, rel * abs(v) * 0.05):
            s, c = out[-1]; out[-1] = (s + v, c + 1)
        else:
            out.append((v, 1))
    return [(s / c, c) for s, c in out]
