# Original code (github.com/amsalort/PINN-eigenvalues) vs pinn_eigen.py

Cloned (depth 1, commit c9a0743 "Add files via upload") to
`.../scratchpad/amsalort/`. Contents: `README.md`, `FBS-PINN-eigenvalues.pdf`, `FBS_PINN_notebook.ipynb`
(155 cells, 9 examples; each cell self-contained, no .py module). It was only read (JSON parsed), never executed.
Cell numbers below are notebook cell indices (0-based) of FBS_PINN_notebook.ipynb.

## Verdict on the p-Laplacian (key finding)
The ORIGINAL code has the SAME issue as pinn_eigen.py. Cell 45 (p=2.2 unit disk; cell 61-65 same for [0,2]^2):

    residual = div - V * u + E_val * u
    loss_residual = (residual ** 2).mean()
    loss_norm = ((u ** 2).mean() - 1) ** 2

i.e. E*u (not E|u|^{p-2}u) and mean(u^2) normalization, although the markdown of cell 44 states the
loss as `\int (\Delta_p u - V|u|^{p-2}u + E|u|^{p-2}u)^2 + \mu(\int |u|^p - 1)^2`. So the published p-Laplacian
tables/figures were produced with the linearised, normalization-dependent residual; text and code disagree.
Also in the original: the p-Laplacian is `div(|grad u|^{p-2} grad u)` with `+1e-12` inside the norm
(same as the repo); V = 0 in both p-Laplacian examples.
Consequence: pinn_eigen.py (before this upgrade) faithfully reproduced the original, so reproducing the
published numbers needs `legacy_plap=True`; the corrected loss (default now) will give different (correct)
values for p != 2 and is what the paper text describes.

## Other comparisons
| item | original notebook | pinn_eigen.py / notebook v2 |
|---|---|---|
| Laplacian loss | `residual = laplace_u - V*u + E_val*u; (residual**2).mean() + mu*((u**2).mean()-1)**2` (cell 9) | identical (means, no |Omega| factor). Now optional `vol`. |
| Ex.1 harmonic potential | code: `return 1.0 * omega**2 * (x**2 + y**2)` (cell 5), markdown (cell 2) says V = 1/2 (x^2+y^2) | `V_harmonic = 0.5*omega**2*|x|^2`. DIFFERENT by factor 2 from the original CODE (equal to the markdown/paper text if paper says 1/2). Must check which the paper's table used: original code = omega^2|x|^2 with omega=1. |
| B(x) | disk `(1 - x**2 - y**2)*u` (cell 3); square `x*(1-x)*y*(1-y)*u` (cell 22); [0,2]^2 `x*(2-x)*y*(2-y)*u`; triangle `x*y*(1-x-y)*u`; 3D/4D ball `(1 - x**2 - y**2 - z**2[-w**2])*u` | same (boundary_disk, boundary_square, boundary_square_2x2, boundary_triangle, boundary_ball) |
| samplers | uniform; ball r=u**(1/3), u**(1/4); triangle by reflection | same |
| mu | Ex1 disk 100 (default, cell 13); square 500 (cell 28: `mu=500`; NOT 3000 as in the paper note/table [44,55]- check); p-disk 4000 (cell 49); p-square `mu=1000 + lam*lam` (cell ~85); triangle 15000; 4D ball 200; 3D ball 400; double-well 400; four-well 1000 | v2 notebook uses the same values (square 500, p-square 1000+E^2 via mu_fn-like loop) |
| p-Laplacian refine | `loss_fn(p, model, xy, E_val, V, mu=1000.0)` in the objective (cell 53) but only `loss_residual` is returned, so mu is irrelevant | repo refine uses mu=0 (equivalent) |
| lr / scheduler | `optim.Adam(model.parameters(), lr=1e-3)`, NO scheduler anywhere (grep: no `scheduler`, no `manual_seed`) | same lr=1e-3, no scheduler; seeds never fixed in the original -> not exactly reproducible |
| epochs / pts / early stop | n_epochs 3000 (disk), 5000 (square), 2000 (p-disk, 3D, 4D, triangle, wells), 2500 (p-square); 5000 pts (triangle 4000, four-well 2000); patience=300, tol=1e-5 | same |
| warm start | `if i>0 and i % K != 0:` copy state_dict else fresh model; K = 50 (disk), 25 (square, p-disk), 20 (p-square), 50 (triangle, 3D), 60 (4D) | same (`reset_every`) |
| meaning of "residual" in tables | `train` returns last-epoch `loss_residual.item()` = mean(residual^2) on the LAST TRAINING BATCH (5000 pts, same points the optimizer just fit; not independent) -> cells 11, 47; refine: `res` of `minimize_scalar` = mean(residual^2) on a fresh 10000-pt sample with mu=0 (cell 17: `xy = sample_points(10000)`; `_, res, _ = loss_fn(..., mu=0.0)`) | same quantities. So "Residual" in tables = squared quantity mean(res^2) ~ Lambda, not r = sqrt(.). Note it is also NOT normalised by mean(u^2) (u normalised to mean(u^2)=1 only approximately, norm_err ~ 1e-3). New: `sweep(return_records=True)` also stores r = sqrt(mean res^2 / mean u^2) on 20000 fresh points. |
| potentials | double well (cell ~112): `V0=1.0; sigma=0.5; a=1; -V0*exp(-0.5*r1sq/sigma**2) - V0*exp(-0.5*r2sq/sigma**2)` with centres (+-1,0,0) = ON the unit-sphere boundary; four-well: `V0=1.0, sigma=0.2, a=0.22`, bump exp(-1/(1-rr^2)) at (+-a,+-a,0), V = -V0*sum | identical to V_double_well_gaussian (a=1,sigma=0.5,V0=1) and V_four_well_bump |
| evaluation of threshold | local min of residual curve below threshold (0.3 disk, 2 square/4D, 2.9 p-disk, 3.5 p-square, 6 triangle, 1.3 3D) | same |

Open point for the coordinator: the original uses mu=500 for the square (cell 28) while CONTEXT lists 3000 for [44,55]; the
exact-value overlay and E range in the original cell are E in [0,105] (80 pts). Check against the paper table.
