"""
Exact and reference eigenvalues used to check the examples.

Every function returns a list of (E, multiplicity, label) sorted by E.

* exact spectra (disk, square, triangle, balls of R^d) are computed from their closed forms;
* the other values are copied from the reference computations of `experiments/`:
  finite elements (P2, curved meshes, Richardson extrapolation) in `experiments/results/fem_summary.json`,
  p-Laplacian references (radial shooting and P1-FEM of nodal subdomains) in
  `experiments/results/plap_refs.json` and `experiments/results/plap_fem_refs.json`.
"""
from __future__ import annotations

from math import comb, pi
from typing import List, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv

Spectrum = List[Tuple[float, int, str]]


def bessel_zeros(nu: float, k: int) -> List[float]:
    """First k positive zeros of J_nu (any real nu >= 0), by sign changes + Brent."""
    zs, x, step = [], max(nu, 0.5) * 0.5 + 0.1, 0.05
    f0 = jv(nu, x)
    while len(zs) < k:
        x1 = x + step
        f1 = jv(nu, x1)
        if f0 == 0.0:
            zs.append(x)
        elif f0 * f1 < 0:
            zs.append(brentq(lambda t: jv(nu, t), x, x1, xtol=1e-14))
        x, f0 = x1, f1
    return zs


def _n_harmonics(m: int, d: int) -> int:
    """Dimension of the spherical harmonics of degree m in R^d."""
    return comb(m + d - 1, d - 1) - (comb(m + d - 3, d - 1) if m >= 2 else 0)


def ball(d: int, E_max: float) -> Spectrum:
    """Dirichlet Laplacian in the unit ball of R^d: E = j_{m+d/2-1,n}^2, multiplicity = #harmonics of degree m."""
    out = []
    m = 0
    while True:
        nu = m + d / 2 - 1
        zs = [z ** 2 for z in bessel_zeros(nu, 12) if z ** 2 <= E_max]
        if not zs:
            break
        out += [(float(E), _n_harmonics(m, d), f"(m,n)=({m},{n + 1})") for n, E in enumerate(zs)]
        m += 1
    return sorted(out)


def disk(E_max: float) -> Spectrum:
    return ball(2, E_max)


def square(E_max: float, a: float = 1.0) -> Spectrum:
    """Dirichlet Laplacian in (0,a)^2: (pi/a)^2 (m^2+n^2)."""
    vals = {}
    for m in range(1, 40):
        for n in range(1, 40):
            E = (pi / a) ** 2 * (m * m + n * n)
            if E <= E_max:
                vals.setdefault(round(E, 9), []).append((m, n))
    return [(E, len(mn), "(m,n)=" + ",".join(f"({m},{n})" for m, n in mn if m <= n))
            for E, mn in sorted(vals.items())]


def triangle(E_max: float) -> Spectrum:
    """Triangle (0,0),(1,0),(0,1): pi^2(m^2+n^2), m > n >= 1 (simple in the windows used here)."""
    out = [(pi ** 2 * (m * m + n * n), 1, f"(m,n)=({m},{n})")
           for m in range(2, 40) for n in range(1, m) if pi ** 2 * (m * m + n * n) <= E_max]
    return sorted(out)


# ── finite element references (experiments/results/fem_summary.json) ──────────────────────────────
ANNULUS = [(39.0133, 1, "FEM"), (40.8725, 2, "FEM"), (46.4285, 2, "FEM"), (55.6179, 2, "FEM")]
HARMONIC_W1 = [(5.8919, 1, "FEM"), (14.8484, 2, "FEM"), (26.5790, 2, "FEM"), (30.6271, 1, "FEM"),
               (40.9385, 2, "FEM"), (49.3852, 2, "FEM")]
HARMONIC_W10 = [(14.2847, 1, "FEM"), (29.1349, 2, "FEM"), (45.0887, 2, "FEM"), (46.2907, 1, "FEM"),
                (62.5793, 2, "FEM"), (66.2805, 2, "FEM")]
DOUBLE_WELL = [(9.5764, 1, "FEM"), (19.7522, 1, "FEM"), (19.9666, 2, "FEM"), (32.7856, 1, "FEM"),
               (32.8735, 2, "FEM"), (33.0339, 2, "FEM"), (39.1924, 1, "FEM"), (48.4061, 1, "FEM"),
               (48.4296, 2, "FEM"), (48.5454, 2, "FEM"), (48.6764, 2, "FEM"), (59.2650, 1, "FEM"),
               (59.4555, 2, "FEM"), (66.5337, 2, "FEM"), (66.5725, 3, "FEM"), (66.7578, 4, "FEM")]
FOUR_WELL = [(9.8542, 1, "FEM"), (20.1801, 2, "FEM"), (20.1888, 1, "FEM"), (33.2104, 2, "FEM"),
             (33.2135, 3, "FEM"), (39.4573, 1, "FEM"), (48.8238, 7, "FEM"), (59.6460, 2, "FEM"),
             (59.6823, 1, "FEM"), (66.9381, 9, "FEM")]

# ── p-Laplacian references (Table tab.plap of the paper; multiplicity is not defined, set to 1) ────
PLAP_DISK_22 = [(6.5317, 1, "radial, positive [shooting]"), (18.3929, 1, "1 nodal diameter [FEM half disk]"),
                (34.8082, 1, "2 nodal diameters [FEM sector pi/2]"), (41.7886, 1, "radial, 1 nodal circle [shooting]"),
                (56.1099, 1, "3 nodal diameters [FEM sector pi/3]")]
PLAP_SQUARE2_22 = [(5.4950, 1, "positive [FEM]"), (15.1429, 1, "diagonal nodal line [FEM]"),
                   (15.2808, 1, "midline nodal line [FEM]"), (25.2487, 1, "2x2 cells [2^p E1]"),
                   (32.4595, 1, "both diagonals nodal [scaled FEM]"), (33.1720, 1, "three strips [FEM]")]
PLAP_BALL3_25 = [(14.1112, 1, "radial, positive [shooting]"), (35.6315, 1, "1 nodal plane [FEM half ball]")]
PLAP_DISK_E1 = {1.5: 4.0178, 2.2: 6.5317, 3.0: 9.8315, 4.0: 14.6816}     # radial shooting


def plap_disk_first(p: float) -> Spectrum:
    return [(PLAP_DISK_E1[p], 1, "radial, positive [shooting]")]
