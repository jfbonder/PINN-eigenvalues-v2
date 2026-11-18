"""
Settings of every example of the paper (manuscript paper/v4/FBS-v4.tex).

One `Example` per figure/table. The settings come from Table `table.settings` and from the text of
Sections 3-5; where the paper does not record a value (marked "our choice" in `notes`) a reasonable
value is used. To change an example, edit its entry here or derive a new one with
`dataclasses.replace(EXAMPLES["disk"], mu=200.0, ...)` and pass it to `examples.core.run_example`.

Vocabulary
----------
windows : list of chains of E nodes. Inside a chain the network is trained node after node; how it is
          initialized at each node is given by `start`. A scan is a single chain; the "three-node
          windows" of Tables 2 and 4 and the FEM windows of Table `table.fem` are several chains,
          each starting from a fresh network.
start   : "warm" (continuation from the previous node; fresh network only at the first node),
          "cold" (fresh network at every node), or an int n (fresh network every n nodes).
mode    : "scan"    -> detect the local minima of the loss curve (below `threshold`, if given),
                       then refine each one with the closed-form quotient of the frozen network;
          "windows" -> in each window take the node of smallest normalized loss
                       (r^2 for p = 2, Lambda_p normalized for p != 2) and refine it.
detect_on : curve used for the detection: "loss_res" (recorded residual term R_mu of the last training
          epoch, as in the figures of Sections 3-4) or "lam_n" (normalized loss on a fresh sample,
          Section 5).
local_h : p-Laplacian only: after detection, a local three-node scan with step local_h around each
          minimum, warm-started from the coarse-scan network (Section 5.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import numpy as np

import pinn_eigen as pe
from examples import references as R


# ── domains ──────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Domain:
    name: str
    dim: int
    sampler: Callable
    boundary: Callable          # B(x), vanishing on the boundary
    vol: float                  # |Omega| (used only by the p-Laplacian loss, see pe.loss_fn)
    box: tuple = (-1.0, 1.0)    # plotting box [a,b]^2 (slice x3=...=0 in dimension >= 3)


def _ball_sampler(d):
    return {3: pe.sample_ball_3d, 4: pe.sample_ball_4d}.get(d, partial(pe.sample_ball, dim=d))


DISK = Domain("unit disk", 2, partial(pe.sample_disk, radius=1.0), pe.boundary_disk, pe.VOL_DISK)
SQUARE = Domain("unit square (0,1)^2", 2, partial(pe.sample_square, x0=0.0, x1=1.0), pe.boundary_square,
                pe.VOL_SQUARE, (0.0, 1.0))
SQUARE2 = Domain("square (0,2)^2", 2, partial(pe.sample_square, x0=0.0, x1=2.0), pe.boundary_square_2x2,
                 pe.VOL_SQUARE_2X2, (0.0, 2.0))
TRIANGLE = Domain("triangle (0,0),(1,0),(0,1)", 2, pe.sample_triangle, pe.boundary_triangle, pe.VOL_TRIANGLE,
                  (0.0, 1.0))
ANNULUS = Domain("annulus 1/2<|x|<1", 2, partial(pe.sample_annulus, r_inner=0.5, r_outer=1.0),
                 partial(pe.boundary_annulus, r_inner=0.5, r_outer=1.0), pe.vol_annulus(0.5, 1.0))


def BALL(d: int) -> Domain:
    return Domain(f"unit ball of R^{d}", d, _ball_sampler(d), pe.boundary_ball, pe.vol_ball(d))


# ── potentials ───────────────────────────────────────────────────────────────────────────────────
V_HARM1 = partial(pe.V_harmonic, omega=1.0)
V_HARM10 = partial(pe.V_harmonic, omega=10.0)
V_DOUBLE = partial(pe.V_double_well_gaussian, a=1.0, sigma=0.5, V0=1.0)
V_FOUR = partial(pe.V_four_well_bump, a=0.22, sigma=0.2, V0=1.0)


# ── grids ────────────────────────────────────────────────────────────────────────────────────────
def grid(a: float, b: float, h: float) -> List[float]:
    """a, a+h, ..., up to b (inclusive up to rounding)."""
    n = int(np.floor((b - a) / h + 1e-9)) + 1
    return [round(a + i * h, 6) for i in range(n)]


def linspace(a: float, b: float, n: int) -> List[float]:
    return [round(float(e), 6) for e in np.linspace(a, b, n)]


def three_nodes(E: float, offsets=(-0.9, 0.1, 1.1)) -> List[float]:
    """Three-node window E_k-0.9, E_k+0.1, E_k+1.1 (Tables 2 and 4)."""
    return [round(E + o, 4) for o in offsets]


def seven_nodes(E: float, half: float = 1.5, h: float = 0.5) -> List[float]:
    """Seven-node window centred at E with step 0.5 (Table table.fem)."""
    k = int(round(half / h))
    return [round(E + (i - k) * h, 6) for i in range(2 * k + 1)]


# ── example ──────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Example:
    name: str
    paper: str                      # figure / table of the paper reproduced
    section: str
    domain: Domain
    windows: List[List[float]]
    mu: float
    n_epochs: int
    V: Callable = pe.V_zero
    potential: str = "V = 0"
    p: float = 2.0
    n_samples: int = 5000
    lr: float = 1e-3
    patience: int = 300             # early stopping: stop if no decrease by tol in `patience` epochs
    tol: float = 1e-5
    lr_decay: bool = False          # cosine decay lr -> 1e-5 over n_epochs (p-Laplacian protocol)
    start: Union[str, int] = "warm"
    mode: str = "scan"
    threshold: Optional[float] = None
    detect_on: str = "loss_res"
    local_h: Optional[float] = None
    hidden: int = 128
    n_hidden: int = 3
    reference: list = field(default_factory=list)   # [(E, multiplicity, label)]
    reference_kind: str = "exact"
    eval_samples: int = 20000       # fresh points for the per-node normalized loss
    cert_chunks: int = 40           # a posteriori quantities on cert_chunks x cert_chunk fresh points
    cert_chunk: int = 10000
    kato_temple: bool = True        # Kato-Temple / Temple using the reference spectrum as the gap
    paper_values: str = ""          # what the paper reports (printed for comparison)
    notes: str = ""

    @property
    def vol(self) -> float:
        """Integration measure in the loss: sample means (vol=1) for p=2, |Omega| for the p-Laplacian."""
        return 1.0 if self.p == 2.0 else self.domain.vol

    def quick(self) -> "Example":
        """Tiny settings for a smoke test (well under a minute on one CPU thread)."""
        wins = self.windows[:2]
        if self.mode == "scan":
            Es = wins[0]
            idx = np.unique(np.linspace(0, len(Es) - 1, min(len(Es), 7)).round().astype(int))
            wins = [[Es[i] for i in idx]]
        # no threshold, so that detection, refinement and plots are exercised
        return replace(self, windows=wins, n_epochs=100, n_samples=500, hidden=32, eval_samples=2000,
                       cert_chunks=4, cert_chunk=1000, threshold=None)


EXAMPLES: Dict[str, Example] = {}


def _add(ex: Example) -> None:
    EXAMPLES[ex.name] = ex


# ═══ Section 3: implementation details ═══════════════════════════════════════════════════════════
_add(Example(
    name="disk", paper="Fig. lossNN, Fig. func", section="3",
    domain=DISK, windows=[grid(0.0, 32.0, 0.5)], mu=100.0, n_epochs=3000, start="warm",
    threshold=1.0, reference=R.disk(33.0),
    paper_values="loss curve with minima at j_{m,n}^2 = 5.7832, 14.6820 (x2), 26.3746 (x2), 30.4713",
    notes="window [0,32] from Table settings; the step is not recorded: h = 0.5 and threshold 1.0 are our choice"))

# ═══ Section 4.1: higher eigenvalues (unit square) ═══════════════════════════════════════════════
_add(Example(
    name="square_44_55", paper="Fig. eigen55", section="4.1",
    domain=SQUARE, windows=[grid(44.0, 55.0, 0.5)], mu=3000.0, n_epochs=3000, start="warm",
    threshold=1.0, reference=R.square(56.0),
    paper_values="one minimum at the double eigenvalue 5 pi^2 = 49.3480",
    notes="step and N_ep not recorded in the paper: h = 0.5, 3000 epochs, warm start are our choice"))
_add(Example(
    name="square", paper="Table 1 (and Fig. collapse)", section="4.1",
    domain=SQUARE, windows=[linspace(0.0, 105.0, 80)], mu=500.0, n_epochs=5000, start=25,
    threshold=2.0, reference=R.square(106.0),
    paper_values="19.739909, 49.348167, 78.956161, 98.695973 (exact 2pi^2, 5pi^2, 8pi^2, 10pi^2); "
                 "r = 0.31, 0.27, 0.44, 0.30",
    notes="80 nodes, re-initialization every 25 nodes (cold starts at E~33, 66, 100 may collapse, "
          "Section 6.1). The paper evaluates the quotient on a 300x300 midpoint grid; here Monte Carlo"))

# ═══ Section 4.2: higher dimensions ══════════════════════════════════════════════════════════════
_add(Example(
    name="ball3", paper="Fig. loss3D (left)", section="4.2",
    domain=BALL(3), windows=[linspace(0.0, 70.0, 100)], mu=100.0, n_epochs=2000, start=15,
    threshold=1.3, reference=R.ball(3, 71.0),
    paper_values="minima near 9.8696, 20.1907 (x3), 33.2175 (x5), 39.4784, 48.83 (x7), 59.68 (x3), 66.95 (x9)",
    notes="window [0,70], mu=100, re-init 15 from Table settings; 100 nodes (h~0.71) and threshold 1.3 are our choice"))
_add(Example(
    name="ball3_windows", paper="Table 2 (d = 3)", section="4.2",
    domain=BALL(3), windows=[three_nodes(E) for E in (9.8696, 20.1907, 33.2175, 39.4784)],
    mu=100.0, n_epochs=2000, start="warm", mode="windows", reference=R.ball(3, 50.0),
    paper_values="9.8757(4) r=0.41; 20.1912(4) r=0.22; 33.2178(4) r=0.26; 39.4789(6) r=0.24",
    notes="fresh network at the first node of each window, warm start at the other two; seed 0"))
_add(Example(
    name="ball4", paper="Fig. loss3D (right)", section="4.2",
    domain=BALL(4), windows=[grid(0.7, 54.7, 1.5)], mu=200.0, n_epochs=2000, start="warm",
    threshold=None, reference=R.ball(4, 56.0),
    paper_values="exactly four local minima at E=14.2, 26.2, 41.2, 50.2 with quotients "
                 "14.6828, 26.3738, 40.7083, 49.2208 (exact 14.6820, 26.3746, 40.7065, 49.2185); "
                 "r = 0.52, 0.38, 0.65, 1.26",
    notes="mirrors experiments/highdim/scan_r4.py (warm start, no re-initialization, all local minima)"))
_add(Example(
    name="ball4_windows", paper="Table 2 (d = 4)", section="4.2",
    domain=BALL(4), windows=[three_nodes(E) for E in (14.6820, 26.3746)],
    mu=200.0, n_epochs=2000, start="warm", mode="windows", reference=R.ball(4, 50.0),
    paper_values="14.6851(4) r=0.49; 26.3751(7) r=0.43; Temple bracket 14.666 <= E1 <= 14.685"))
_add(Example(
    name="ball10", paper="Section 4.2 (ball of R^10)", section="4.2",
    domain=BALL(10), windows=[grid(54.0, 84.0, 3.0)], mu=2000.0, n_epochs=2000, start="warm",
    threshold=100.0, reference=R.ball(10, 90.0),
    paper_values="two wells (19.8 at E=57, 49.4 at E=78); quotients 57.62-57.84 (exact 57.583) and "
                 "77.10-77.70 (exact 76.939, multiplicity 10); r ~ 4.7-7",
    notes="mirrors experiments/highdim/run_d10.py with mu=2000 (mu=200 collapses at a cold start)"))

# ═══ Section 4.3: other geometries ══════════════════════════════════════════════════════════════
_add(Example(
    name="triangle", paper="Fig. loss.lap.triang", section="4.3",
    domain=TRIANGLE, windows=[linspace(0.0, 180.0, 100)], mu=15000.0, n_epochs=2000, n_samples=4000,
    start=50, threshold=6.0, reference=R.triangle(181.0),
    paper_values="minima near 5pi^2, 10pi^2, 13pi^2, 17pi^2 = 49.35, 98.70, 128.30, 167.78",
    notes="settings of Table settings (h=1.8, re-init 50, N=4000); threshold 6 as in the earlier notebook"))
_add(Example(
    name="triangle_windows", paper="Table 4", section="4.3",
    domain=TRIANGLE, windows=[three_nodes(E) for E in (5 * np.pi ** 2, 10 * np.pi ** 2, 13 * np.pi ** 2,
                                                         17 * np.pi ** 2)],
    mu=15000.0, n_epochs=2000, n_samples=4000, start="warm", mode="windows", reference=R.triangle(181.0),
    paper_values="49.3509(14) r=0.94; 98.7067(24) r=1.81; 128.4612(95) r=5.68; 167.7923(35) r=2.51 "
                 "(the cold first node of every window did not converge)"))
_add(Example(
    name="annulus", paper="Fig. loss.lap.annulus", section="4.3",
    domain=ANNULUS, windows=[grid(30.0, 56.0, 1.0)], mu=200.0, n_epochs=3000, start="warm",
    threshold=2.0, reference=R.ANNULUS, reference_kind="FEM (exact to 1e-6)",
    paper_values="the warm-started network stayed on the first eigenvalue 39.0133; 40.8725 (x2) and "
                 "46.4285 (x2) were missed",
    notes="the settings of the paper's annulus run were not recorded; window [30,56], h=1, mu=200, "
          "3000 epochs are our choice (mu >= D^2 = 81 on this window)"))

# ═══ Section 4.4: other potentials ═══════════════════════════════════════════════════════════════
_add(Example(
    name="harmonic_w1", paper="Fig. pot2", section="4.4",
    domain=DISK, V=V_HARM1, potential="harmonic, omega = 1", windows=[linspace(0.0, 55.0, 80)],
    mu=100.0, n_epochs=3000, start=50, threshold=1.0, reference=R.HARMONIC_W1, reference_kind="FEM",
    paper_values="FEM eigenvalues 5.8919, 14.8484 (x2), 26.5790 (x2), 30.6271, 40.9385 (x2), 49.3852 (x2); "
                 "the minima were not refined in the paper"))
_add(Example(
    name="harmonic_w10", paper="Fig. pot1", section="4.4",
    domain=DISK, V=V_HARM10, potential="harmonic, omega = 10", windows=[linspace(0.0, 55.0, 80)],
    mu=100.0, n_epochs=3000, start=50, threshold=10.0, reference=R.HARMONIC_W10, reference_kind="FEM",
    paper_values="minima near 14.2847, 29.1349 (x2), one minimum for 45.0887 (x2)/46.2907, spurious "
                 "minimum at E~49.4; at E=0 the network collapses to u=0 (regime iv)"))
_add(Example(
    name="double_well", paper="Fig. well-two (left)", section="4.4",
    domain=BALL(3), V=V_DOUBLE, potential="double Gaussian well, a=1, sigma=0.5, V0=1",
    windows=[linspace(0.0, 75.0, 120)], mu=400.0, n_epochs=2000, start=50, threshold=1.3,
    reference=R.DOUBLE_WELL, reference_kind="FEM",
    paper_values="loss curve; FEM eigenvalues in clusters (Section 4.5)"))
_add(Example(
    name="four_well", paper="Fig. well-four (right)", section="4.4",
    domain=BALL(3), V=V_FOUR, potential="four compactly supported wells, a=0.22, sigma=0.2",
    windows=[linspace(0.0, 75.0, 120)], mu=1000.0, n_epochs=2000, n_samples=2000, start=50, threshold=1.3,
    reference=R.FOUR_WELL, reference_kind="FEM",
    paper_values="loss curve; FEM eigenvalues in clusters (Section 4.5)"))

# ═══ Section 4.5: validation with finite elements ════════════════════════════════════════════════
_add(Example(
    name="double_well_windows", paper="Table fem (double well), Fig. fem (top)", section="4.5",
    domain=BALL(3), V=V_DOUBLE, potential="double Gaussian well",
    windows=[seven_nodes(E) for E in (9.5764, 19.7522, 32.8735, 39.1924)],
    mu=400.0, n_epochs=2000, start="warm", mode="windows", reference=R.DOUBLE_WELL, reference_kind="FEM",
    kato_temple=False,
    paper_values="seed 0: 9.5783, 19.7640, 32.9552, 39.1944 (FEM 9.5764, 19.7522, 32.8735, 39.1924)",
    notes="windows centred at the FEM values (accuracy test, not detection); the clusters make a "
          "Kato-Temple gap meaningless, so it is not computed"))
_add(Example(
    name="four_well_windows", paper="Table fem (four well), Fig. fem (middle)", section="4.5",
    domain=BALL(3), V=V_FOUR, potential="four compactly supported wells",
    windows=[seven_nodes(E) for E in (9.8542, 20.1801, 33.2104, 39.4573)],
    mu=1000.0, n_epochs=2000, n_samples=2000, start="warm", mode="windows", reference=R.FOUR_WELL,
    reference_kind="FEM", kato_temple=False,
    paper_values="seed 0: 9.8593, 20.1505, 33.2141, 39.4578 (FEM 9.8542, 20.1801, 33.2104, 39.4573)"))
_add(Example(
    name="harmonic_w10_windows", paper="Table fem (harmonic omega=10), Fig. fem (bottom)", section="4.5",
    domain=DISK, V=V_HARM10, potential="harmonic, omega = 10",
    windows=[seven_nodes(E) for E in (14.2847, 29.1349, 45.0887, 62.5793)],
    mu=100.0, n_epochs=3000, start="warm", mode="windows", reference=R.HARMONIC_W10, reference_kind="FEM",
    paper_values="seed 0: 14.2899, 29.1357, 45.0892, 62.5792"))
_add(Example(
    name="double_well_blind", paper="Table blind", section="4.5",
    domain=BALL(3), V=V_DOUBLE, potential="double Gaussian well",
    windows=[grid(4.7, 49.7, 1.5)], mu=400.0, n_epochs=2000, start="warm", threshold=None,
    reference=R.DOUBLE_WELL, reference_kind="FEM", kato_temple=False,
    paper_values="five local minima at E=9.2, 19.7, 33.2, 39.2, 48.2 with quotients 9.5779, 19.7545, "
                 "32.7966, 39.1945, 48.5270 (one per FEM cluster)"))

# ═══ Section 5: p-Laplacian (homogeneous loss, protocol P1) ═════════════════════════════════════
_P1 = dict(mu=200.0, n_epochs=1500, patience=10 ** 9, lr_decay=True, start="warm", eval_samples=40000,
           detect_on="lam_n", kato_temple=False, reference_kind="shooting / P1-FEM")
_add(Example(
    name="plap_disk", paper="Table plap (disk), Fig. plap (left)", section="5", p=2.2,
    domain=DISK, windows=[grid(3.0, 58.0, 1.0)], local_h=0.5, reference=R.PLAP_DISK_22, **_P1,
    paper_values="E_PINN 6.5337, 18.4069, 34.8043, 41.9615, 56.1166 "
                 "(refs 6.5317, 18.3929, 34.8082, 41.7886, 56.1099)",
    notes="the paper ran the window as two warm-started pieces [3,25] and [24,58]; here one chain"))
_add(Example(
    name="plap_square", paper="Table plap (square), Fig. plap (centre)", section="5", p=2.2,
    domain=SQUARE2, windows=[grid(3.0, 37.0, 1.0)], local_h=0.5, reference=R.PLAP_SQUARE2_22, **_P1,
    paper_values="E_PINN 5.4902, 15.1440, 25.2285, 32.4615 (refs 5.4950, 15.1429, 25.2487, 32.4595)",
    notes="the paper ran the window as two warm-started pieces [3,20] and [19,37]; here one chain"))
_add(Example(
    name="plap_ball3", paper="Table plap (ball of R^3), Fig. plap (right)", section="5", p=2.5,
    domain=BALL(3), windows=[grid(8.0, 46.0, 2.0)], local_h=1.0, reference=R.PLAP_BALL3_25, **_P1,
    paper_values="E_PINN 14.1004, 35.6283 (refs 14.1112, 35.6315)"))
_add(Example(
    name="plap_disk_p3", paper="Table plap.p (p = 3)", section="5", p=3.0,
    domain=DISK, windows=[[8.5, 9.25, 10.0, 10.75, 11.5]], mode="windows",
    reference=R.plap_disk_first(3.0), **_P1,
    paper_values="E_PINN at the grid minimum 9.801 (seed 0); ref 9.8315"))
_add(Example(
    name="plap_disk_p4", paper="Table plap.p (p = 4)", section="5", p=4.0,
    domain=DISK, windows=[[13.0, 14.0, 15.0, 16.0, 17.0]], mode="windows",
    reference=R.plap_disk_first(4.0), **_P1,
    paper_values="E_PINN at the grid minimum 14.895 (seed 0); ref 14.6816 (p = 4 is marginal)"))
