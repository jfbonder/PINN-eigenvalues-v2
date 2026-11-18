# Independent finite-element validation (Reviewer #1, round 2, point 2)

Everything here is reproducible from `experiments/fem/` (1 CPU core). Pipeline:
`run_fem.py` (FEM, scikit-fem 12.0.2) -> `fem_analysis.py` (Richardson extrapolation, exact references, multiplicities)
-> `pinn_windows.py` (PINN local sweeps, resumable) -> `make_report.py` (figures, `tables.tex`, `results_tables.md`).
Raw results: `experiments/results/fem_*.json`, `fem_pinn_*_s<seed>.json`, `fem_summary.json`, `fem_pinn_summary.json`; logs in `logs/`.

## 1. Problems (pinned down from the v3 source, `pinn_eigen.py`, original notebook)
All problems are `-Delta u + V u = E u` in Omega, `u = 0` on the boundary (Dirichlet; the "potentials" are NOT on R^n).

| example | Omega | V | notes |
|---|---|---|---|
| double well | unit ball in R^3 | `-[exp(-|x-e1|^2/(2 s^2)) + exp(-|x+e1|^2/(2 s^2))]`, `s=0.5`, V0=1, a=1 (= `-e^{-2|x-x0|^2}-e^{-2|x+x0|^2}`, x0=(1,0,0) as in paper text, consistent with code) | well centres lie ON the boundary sphere; axisymmetric about the x-axis |
| four well | unit ball in R^3 | `-sum_i phi(|x-c_i|/0.2)`, `phi(t)=exp(-1/(1-t^2))` for t<1, c_i=(+-0.22,+-0.22,0) | compact support, max depth e^{-1}=0.37 (not 1) |
| harmonic | unit disk | `V=(w^2/2)|x|^2`, w=1, 10 | **see discrepancy D1** |
| annulus | 0.5<|x|<1 | 0 | |
| square / triangle | [0,1]^2 / (0,0),(1,0),(0,1) | 0 | sanity |

## 2. FEM method
Lagrange P2 on **curved (isoparametric) meshes** (`MeshTri2.init_circle`, `MeshTet2.init_ball`, polar annulus mesh with
boundary edge midpoints projected to the circles; straight-sided P2 for square/triangle). Stiffness + potential-mass and mass matrices
(quadrature order 6, so the compactly supported bump is integrated accurately), generalized problem solved with
`scipy.sparse.linalg.eigsh` shift-invert (SuperLU). Reference values: Richardson extrapolation from the last three
uniform refinements, observed order `p` estimated per eigenvalue (clipped to [1.5, 4.5]); the *estimated error* reported is
`|E_extrap - E_finest|` (conservative: on the free ball the true error of the extrapolated value is 3-10x smaller than this estimate).
Multiplicities: eigenvalues of the finest mesh are grouped if their gap is below 0.1 x (change between the last two levels) + 1e-4
(mesh asymmetry splits true multiplets by an amount of the order of the discretisation error; cluster sizes agree with the exact ones for all validation cases).

## 3. Validation against exact values and mesh convergence
2-D: observed order **p = 4.0** (disk, annulus, square, triangle, harmonic) – errors 1e-10..1e-6 for E<60 on 32k dof.
3-D ball (curved P2 tets, only 3 levels: 575 / 4 991 / 41 727 dof; level 5 does not fit in 15 GB with a direct solver): observed **p ~ 2.7-2.9**
(mesh quality of `init_ball`, not the P2 theory limit). Extrapolated errors vs exact j^2_{l+1/2,n}: 7e-5 (pi^2), 7e-4 (20.19), 2.4e-3 (33.22), 1.8e-3 (39.48), 6e-3 (48.83), 1.6e-2 (66.95).
Harmonic references were cross-checked against the audit's independent radial finite-volume reference (agreement 1e-7).
Figure: `figures/fem_convergence.png`.

