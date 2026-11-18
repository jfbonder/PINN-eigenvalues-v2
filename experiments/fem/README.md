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

## 4. FEM reference eigenvalues (tables in the appendix; LaTeX in `tables.tex`)
Spectra of the free ball and the two wells: `figures/fem_spectra_ball.png`. Key values (extrapolated, with multiplicity):
- **double well**: 9.5764(1), 19.7522(1), 19.9666(2), 32.786(1), 32.874(2), 33.034(2), 39.192(1), 48.41-48.68 (7 eigenvalues within 0.27), 59.26(1), 59.46(2), 66.53(2)...; est. errors 5e-4 (E1) to 1e-2 (E~33-39), up to 0.1 for E>48.
- **four well**: 9.8542(1), 20.180(2), 20.189(1), 33.210(2)+33.214(3), 39.457(1), 48.82(7 within 0.1), 59.65(2), 59.68(1)...; free ball: 9.8696, 20.1907(3), 33.2175(5), 39.4784, 48.8312(7).
- **harmonic disk w=1** (V=w^2/2 r^2): 5.8919(1), 14.8484(2), 26.5790(2), 30.6271(1), 40.9385(2), 49.3852(2). **w=10**: 14.2847(1), 29.1349(2), 45.0887(2), 46.2907(1), 62.5793(2), 66.2805(2).
- **annulus 0.5<|x|<1**: 39.0133(1), 40.8725(2), 46.4285(2), 55.6179(2).
- square: pi^2(m^2+n^2) reproduced to 4e-8 (E<=98.7); triangle: 49.34802, 98.69604 (both simple), 128.305, 167.784, 197.393.

## 5. PINN vs FEM
Setup: paper/notebook settings (3x128 tanh, Adam lr 1e-3, 5000 MC points (2000 for the four well), up to 2000 epochs (3000 for the disk), early stopping patience 300/tol 1e-5,
mu = 400 / 1000 / 100 for double well / four well / harmonic, warm start along E, fresh MC batch per epoch, means instead of integrals, 1 CPU thread). For each example,
4 windows of 7 grid points (step 0.5, half-width 1.5, i.e. finer than the paper's step ~0.65-0.7) were swept, **windows were centred using the FEM values** (that is the point of the experiment: how
well the loss curve reproduces known eigenvalues; it is not a blind search). The first point of a window starts from a fresh network, the rest are warm-started.
Seeds: 0,1 (all), 2 (four well only; the container restart interrupted the harmonic seed 2, so harmonic and double well have 2 seeds). Per E-point the fresh-sample (20000 pts) normalised
loss `Lambda = mean(res^2)/mean(u^2)`, the recorded training residual term (what the paper plots), and Rayleigh quotients of the trained network (4e5 fixed MC points; 1e5 for double-well seed 0 only, which is why its two quotients differ by up to 0.5 - MC noise; the quotient `-int u(Lap u - V u)/int u^2` is used throughout) are stored.
Three estimators per window: (rho) Rayleigh quotient of the network at the grid node of smallest normalised loss (this is the `refine` step of the paper, done in closed form); (par) 3-point parabolic
fit (theory_notes, Prop. parabola) of the normalised loss; (par-rec) the same on the recorded loss.

PINN vs FEM table (mean over seeds, [min,max]); relative errors are vs the nearest FEM eigenvalue / cluster member (`#FEM within 0.75` > 1 flags an unresolved cluster):

| case | window centre | FEM ref (mult) | #FEM eigenvalues within 0.75 | PINN Rayleigh quotient (mean over seeds [min,max]) | rel.err | parabola on normalised loss | rel.err | parabola on recorded loss (paper-style) | rel.err | min of Lambda | time/E-point |
|---|---|---|---|---|---|---|---|---|---|---|---|
| double_well | 9.58 | 9.5764±5e-04 (1) | 1 | 9.5782 [9.5782,9.5783] | +0.019% | 9.5717 [9.5496,9.5937] | -0.05% | 9.6190 [9.5684,9.6696] | +0.44% | 0.065 | 197 s |
| double_well | 19.86 | 19.7522±3e-03 (1) | 2 | 19.7593 [19.7547,19.7640] | +0.036% | 19.7341 [19.7052,19.7630] | -0.09% | 19.6846 [19.6459,19.7232] | -0.34% | 0.077 | 231 s |
| double_well | 32.90 | 32.8735±2e-02 (2) | 3 | 32.9062 [32.8571,32.9552] | +0.099% | 32.9228 [32.8972,32.9483] | +0.15% | 32.9586 [32.9180,32.9992] | +0.26% | 0.117 | 182 s |
| double_well | 39.20 | 39.1924±1e-02 (1) | 1 | 39.1941 [39.1938,39.1944] | +0.004% | 39.2305 [39.1954,39.2655] | +0.10% | 39.1894 [39.1011,39.2777] | -0.01% | 0.108 | 174 s |
| four_well | 9.85 | 9.8542±5e-04 (1) | 1 | 9.8606 [9.8581,9.8646] | +0.066% | 9.8643 [9.8338,9.8871] | +0.10% | 9.7957 [9.7760,9.8223] | -0.59% | 0.355 | 74 s |
| four_well | 20.19 | 20.1801±3e-03 (2) | 2 | 20.1751 [20.1505,20.1878] | -0.025% | 20.3348 [20.1062,20.6807] | +0.77% | 19.9790 [19.8362,20.2386] | -1.00% | 0.478 | 65 s |
| four_well | 33.23 | 33.2104±1e-02 (2) | 2 | 33.2093 [33.2061,33.2141] | -0.003% | 33.1519 [32.9924,33.2441] | -0.18% | 33.1973 [33.0519,33.3717] | -0.04% | 0.407 | 66 s |
| four_well | 39.47 | 39.4573±1e-02 (1) | 1 | 39.4515 [39.4414,39.4578] | -0.015% | 39.2486 [38.9358,39.8181] | -0.53% | 39.3994 [39.0767,39.8865] | -0.15% | 1.054 | 71 s |
| harm_w10 | 14.28 | 14.2847±3e-07 (1) | 1 | 14.2901 [14.2899,14.2904] | +0.038% | 14.3640 [14.3351,14.3929] | +0.55% | 14.4610 [14.3547,14.5673] | +1.23% | 0.174 | 110 s |
| harm_w10 | 29.13 | 29.1349±2e-06 (2) | 1 | 29.1355 [29.1353,29.1357] | +0.002% | 29.1433 [29.1251,29.1616] | +0.03% | 29.1851 [29.1753,29.1948] | +0.17% | 0.056 | 115 s |
| harm_w10 | 45.70 | 45.0887±5e-06 (2) | 1 | 45.0892 [45.0892,45.0892] | +0.001% | 45.0959 [45.0952,45.0965] | +0.02% | 45.1012 [45.0996,45.1028] | +0.03% | 0.065 | 149 s |
| harm_w10 | 62.58 | 62.5793±2e-05 (2) | 1 | 62.5796 [62.5792,62.5800] | +0.000% | 62.6267 [62.6249,62.6285] | +0.08% | 62.6373 [62.6265,62.6480] | +0.09% | 0.076 | 195 s |

Figures: `figures/pinn_vs_fem_{double_well,four_well,harm_w10}.png` (loss curves of the windows with FEM eigenvalues as red dashed lines and the exact curve min_k (E-E_k)^2 dotted).

Findings.
1. **Rayleigh-quotient estimates agree with FEM to <= 0.1% in all isolated windows** (typically 1e-4..5e-4; 0.002%-0.04% for harmonic w=10; 9.5782 vs 9.5764 for the double-well ground state). Parabolic fits on the loss curve are 5-50x worse (0.05-1.3%, worst in the four-well window at 39.47 where the loss floor is high, min Lambda ~1) because the loss curve is flat-bottomed/noisy with floor 0.05-1; the paper-style grid minimum is only accurate to ~step/2 = 0.25.
2. **Zero of the loss is not attained**: min of Lambda over the window is 0.06-0.12 (double well, harmonic) and 0.35-1.1 (four well, mu=1000, 2000 points), i.e. residual r = 0.25-1: PINN eigenvalue accuracy comes from the quotient (quadratic in r), not from the depth of the minimum.
3. **Clusters are not resolved**: double-well 19.75 | 19.97(x2): PINN finds a single minimum with quotient 19.76 (the lower one); 32.79|32.87(x2)|33.03(x2): quotient 32.91 (inside the cluster, within 0.12 of all members); four-well 20.18(x2)+20.19 and 33.21(x2)+33.21(x3) are within the mesh error and are naturally single minima. The multiplicity cannot be read from the curve.
4. **The well potentials are weak perturbations**: E1 shifts by -0.293 (double well) and only -0.0154 (four well) relative to the free ball (pi^2); the four-well shift is of the same size as the PINN error (+0.006..0.008), so the four-well example barely distinguishes the potential from V=0 (consistent with audit C-11).
5. **Missed eigenvalue, harmonic w=10**: the window around 45.7 contains the double eigenvalue 45.0887 and the simple 46.2907; the warm-started PINN stays on the 45.09 branch (loss peak at 46.7 instead of a dip at 46.29) -> 46.29 is not found; the hysteresis of warm starts hides close eigenvalues (gap 1.2 vs step 0.5). Same mechanism as audit finding 8.
6. Cost (1 core, this container, load from other jobs, so +-50%): PINN 197 s (double well), 69 s (four well), 142 s (w=10 disk) per E-point on average (early stopping: 840-1050 epochs on average); a 7-point window 7-25 min, a paper-style 120-point sweep of [0,75] would take ~7 h (double well). FEM: <10 s in 2-D (32k dof) and 112-134 s for the 3-D ball (42k dof, 60 eigenpairs, all eigenvalues in [0,75] with multiplicities, error 1e-3..1e-1).
   FEM is therefore 2-3 orders of magnitude cheaper and more accurate for these linear problems (as the paper itself says: the method is not meant to compete with FEM on linear problems).

## 6. Discrepancies with the paper (v3)
- **D1 harmonic potential**: the paper text/code `V_harmonic` use V=(w^2/2)|x|^2; the original notebook code used w^2|x|^2. FEM: (w^2/2): 5.8919 (w=1), 14.2847 (w=10); (w^2): 6.0000 (w=1), 20.0161 (w=10). The figures' minima (w=10: ~14.3, ~29, ~45, ~62) and the PINN values (14.29) match the w^2/2 version; the paper's text is right and the original notebook code is not what produced the figures. (The values 5.8931/14.2944 quoted in the brief are within 0.02%/0.07% of FEM 5.8919/14.2847.)
- **D2 annulus (Fig. 9)**: confirmed: eigenvalues in [0,50] are 39.0133 (1), 40.8725 (2), 46.4285 (2): five eigenfunctions, three distinct values; the paper's theoretical curve has one zero and the PINN curve misses two.
- **D3 harmonic w=10 (Fig. 12)**: 45.0887 (x2) and 46.2907 are not resolved (see finding 5).
- **D4 wells**: double well centres sit on the boundary; E1=9.5764 vs 9.8696 free; four well E1=9.8542, practically free ball. No eigenvalues are reported in the paper for these cases; the table above can be added. The paper's loss curves (Figs. 17-18) show single minima at ~9.6, 19.9, 33, 39.3, 48.6, 59.5, 66.7: consistent with FEM cluster positions but each of the later minima stands for 3-9 eigenvalues.
- Four-well depth is e^{-1}=0.37 per well, not 1 (V0 multiplies phi whose max is e^{-1}).

## 7. Limitations
3-D FEM has only 3 mesh levels with observed order ~2.8, error estimates are of Richardson type (verified conservative on the free ball). PINN windows were centred with FEM knowledge and 2-3 seeds only; the extra seeds of the harmonic and double-well cases were cut for time (container restart). Model weights for runs after the first double-well seed are in `experiments/results/fem_pinn_models/`.


## Appendix: all generated tables
<!-- generated by make_report.py -->
### Validation
| problem | mult | exact / reference | FEM extrapolated | abs. error | estimated error | obs. order p |
|---|---|---|---|---|---|---|
| unit disk (j_mn^2) | 1 | 5.783186 | 5.783186 | 2.0e-10 | 3.2e-08 | 3.99 |
|  | 2 | 14.681971 | 14.681971 | 5.7e-09 | 5.2e-07 | 3.98 |
|  | 2 | 26.374616 | 26.374616 | 4.5e-08 | 3.3e-06 | 3.97 |
|  | 1 | 30.471262 | 30.471262 | 6.5e-08 | 4.5e-06 | 3.97 |
|  | 2 | 40.706466 | 40.706466 | 2.1e-07 | 1.3e-05 | 3.97 |
|  | 2 | 49.218456 | 49.218456 | 3.1e-07 | 2.0e-05 | 3.97 |
| annulus 0.5<r<1 (cross-product zeros) | 1 | 39.013288 | 39.013288 | 1.8e-10 | 3.4e-07 | 4.00 |
|  | 2 | 40.872453 | 40.872453 | 1.5e-10 | 4.6e-07 | 4.00 |
|  | 2 | 46.428454 | 46.428454 | 1.3e-09 | 8.5e-07 | 3.99 |
|  | 2 | 55.617888 | 55.617888 | 3.7e-09 | 1.6e-06 | 3.99 |
| unit ball (j_{l+1/2,n}^2) | 1 | 9.869604 | 9.869538 | 6.7e-05 | 4.8e-04 | 2.87 |
|  | 3 | 20.190729 | 20.190035 | 6.9e-04 | 4.6e-03 | 2.78 |
|  | 5 | 33.217462 | 33.215040 | 2.4e-03 | 2.0e-02 | 2.70 |
|  | 1 | 39.478418 | 39.476648 | 1.8e-03 | 1.4e-02 | 2.96 |
|  | 7 | 48.831194 | 48.824949 | 6.2e-03 | 6.9e-02 | 2.68 |
|  | 3 | 59.679516 | 59.679126 | 3.9e-04 | 5.7e-02 | 2.93 |
|  | 9 | 66.954312 | 66.938587 | 1.6e-02 | 2.1e-01 | 2.70 |
| unit square | 1 | 19.739209 | 19.739209 | 2.0e-10 | 9.4e-08 | 3.99 |
|  | 2 | 49.348022 | 49.348022 | 4.2e-09 | 9.2e-07 | 3.99 |
|  | 1 | 78.956835 | 78.956835 | 2.1e-08 | 4.5e-06 | 3.99 |
|  | 2 | 98.696044 | 98.696044 | 4.5e-08 | 8.0e-06 | 3.99 |
| right triangle | 1 | 49.348022 | 49.348022 | 1.2e-07 | 1.1e-05 | 3.98 |
|  | 1 | 98.696044 | 98.696042 | 1.8e-06 | 1.0e-04 | 3.97 |
| harmonic w=1 (radial FD ref.) | 1 | 5.891903 | 5.891903 | 4.7e-09 | 3.2e-08 | 3.99 |
|  | 2 | 14.848397 | 14.848397 | 5.5e-09 | 5.2e-07 | 3.98 |
|  | 2 | 26.579022 | 26.579022 | 5.8e-08 | 3.3e-06 | 3.97 |
|  | 1 | 30.627080 | 30.627080 | 6.0e-08 | 4.4e-06 | 3.97 |
|  | 2 | 40.938510 | 40.938510 | 2.8e-07 | 1.3e-05 | 3.97 |
|  | 2 | 49.385181 | 49.385180 | 3.1e-07 | 2.0e-05 | 3.97 |
| harmonic w=10 (radial FD ref.) | 1 | 14.284741 | 14.284741 | 7.4e-09 | 3.5e-07 | 3.97 |
|  | 2 | 29.134944 | 29.134944 | 2.4e-08 | 1.7e-06 | 3.98 |
|  | 2 | 45.088748 | 45.088748 | 7.7e-08 | 5.3e-06 | 3.97 |
|  | 1 | 46.290734 | 46.290734 | 1.1e-07 | 5.6e-06 | 3.95 |
|  | 2 | 62.579331 | 62.579331 | 2.0e-07 | 1.7e-05 | 3.97 |
|  | 2 | 66.280485 | 66.280484 | 3.3e-07 | 2.4e-05 | 3.97 |

### Mesh levels: ball, free
| level | dof | elements | wall (s) | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 575 | 512 | 0.1 | 9.89528 | 20.32351 | 20.33656 | 20.36778 | 33.66312 | 33.74476 |
| 3 | 4991 | 4096 | 3.7 | 9.87305 | 20.20943 | 20.20958 | 20.21844 | 33.26631 | 33.29537 |
| 4 | 41727 | 32768 | 123.5 | 9.87002 | 20.19285 | 20.19291 | 20.19433 | 33.22217 | 33.22639 |

### Mesh levels: double well
| level | dof | elements | wall (s) | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 575 | 512 | 0.1 | 9.60253 | 19.90108 | 20.10443 | 20.13895 | 33.30020 | 33.42171 |
| 3 | 4991 | 4096 | 3.0 | 9.58002 | 19.77169 | 19.98643 | 19.99525 | 32.85747 | 32.96018 |
| 4 | 41727 | 32768 | 112.4 | 9.57694 | 19.75472 | 19.96968 | 19.97117 | 32.79561 | 32.88838 |

### Mesh levels: four well
| level | dof | elements | wall (s) | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 575 | 512 | 0.1 | 9.88033 | 20.31616 | 20.32691 | 20.36428 | 33.66165 | 33.73867 |
| 3 | 4991 | 4096 | 2.1 | 9.85768 | 20.19944 | 20.19959 | 20.21727 | 33.26478 | 33.28781 |
| 4 | 41727 | 32768 | 133.5 | 9.85464 | 20.18275 | 20.18282 | 20.19342 | 33.21796 | 33.22074 |

### Mesh levels: harmonic w=10
| level | dof | elements | wall (s) | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 481 | 256 | 0.1 | 14.28602 | 29.14148 | 29.14148 | 45.10619 | 45.10832 | 46.31042 |
| 4 | 1985 | 1024 | 0.3 | 14.28483 | 29.13537 | 29.13537 | 45.08991 | 45.09006 | 46.29208 |
| 5 | 8065 | 4096 | 1.6 | 14.28475 | 29.13497 | 29.13497 | 45.08882 | 45.08883 | 46.29082 |
| 6 | 32513 | 16384 | 7.5 | 14.28474 | 29.13495 | 29.13495 | 45.08875 | 45.08875 | 46.29074 |

### Mesh levels: annulus
| level | dof | elements | wall (s) | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|---|---|---|
| 2 | 2250 | 1200 | 0.1 | 39.01470 | 40.87432 | 40.87432 | 46.43186 | 46.43186 |
| 3 | 9362 | 4832 | 0.4 | 39.01338 | 40.87257 | 40.87257 | 46.42867 | 46.42867 |
| 4 | 38052 | 19328 | 2.4 | 39.01329 | 40.87246 | 40.87246 | 46.42847 | 46.42847 |
| 5 | 153162 | 77184 | 19.8 | 39.01329 | 40.87245 | 40.87245 | 46.42846 | 46.42846 |

### Reference eigenvalues: double_well, E<=75
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 9.5764 | 4.9e-04 | 1 | 0.0e+00 |
| 2 | 19.7522 | 2.6e-03 | 1 | 0.0e+00 |
| 3-4 | 19.9666 | 4.8e-03 | 2 | 1.5e-03 |
| 5 | 32.7856 | 1.0e-02 | 1 | 0.0e+00 |
| 6-7 | 32.8735 | 2.3e-02 | 2 | 6.4e-03 |
| 8-9 | 33.0339 | 1.5e-02 | 2 | 9.6e-04 |
| 10 | 39.1924 | 1.4e-02 | 1 | 0.0e+00 |
| 11 | 48.4061 | 3.1e-02 | 1 | 0.0e+00 |
| 12-13 | 48.4296 | 9.0e-02 | 2 | 3.0e-02 |
| 14-15 | 48.5454 | 6.6e-02 | 2 | 2.6e-03 |
| 16-17 | 48.6764 | 5.4e-02 | 2 | 4.7e-03 |
| 18 | 59.2650 | 3.9e-02 | 1 | 0.0e+00 |
| 19-20 | 59.4555 | 5.7e-02 | 2 | 3.0e-02 |
| 21-22 | 66.5337 | 7.4e-02 | 2 | 5.3e-03 |
| 23-25 | 66.5725 | 2.0e-01 | 3 | 6.4e-02 |
| 26-29 | 66.7578 | 1.9e-01 | 4 | 1.1e-01 |

### Reference eigenvalues: four_well, E<=75
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 9.8542 | 4.7e-04 | 1 | 0.0e+00 |
| 2-3 | 20.1801 | 2.8e-03 | 2 | 6.3e-05 |
| 4 | 20.1888 | 4.6e-03 | 1 | 0.0e+00 |
| 5-6 | 33.2104 | 1.2e-02 | 2 | 2.8e-03 |
| 7-9 | 33.2135 | 2.0e-02 | 3 | 3.7e-03 |
| 10 | 39.4573 | 1.4e-02 | 1 | 0.0e+00 |
| 11-17 | 48.8238 | 6.9e-02 | 7 | 5.5e-02 |
| 18-19 | 59.6460 | 3.8e-02 | 2 | 4.6e-04 |
| 20 | 59.6823 | 5.7e-02 | 1 | 0.0e+00 |
| 21-29 | 66.9381 | 2.1e-01 | 9 | 1.9e-01 |

### Reference eigenvalues: harm_w1, E<=55
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 5.8919 | 3.2e-08 | 1 | 0.0e+00 |
| 2-3 | 14.8484 | 5.2e-07 | 2 | 1.7e-13 |
| 4-5 | 26.5790 | 3.3e-06 | 2 | 3.3e-07 |
| 6 | 30.6271 | 4.4e-06 | 1 | 0.0e+00 |
| 7-8 | 40.9385 | 1.3e-05 | 2 | 1.4e-13 |
| 9-10 | 49.3852 | 2.0e-05 | 2 | 2.5e-13 |

### Reference eigenvalues: harm_w10, E<=55
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 14.2847 | 3.5e-07 | 1 | 0.0e+00 |
| 2-3 | 29.1349 | 1.7e-06 | 2 | 2.7e-13 |
| 4-5 | 45.0887 | 5.3e-06 | 2 | 5.8e-07 |
| 6 | 46.2907 | 5.6e-06 | 1 | 0.0e+00 |

### Reference eigenvalues: annulus, E<=50
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 39.0133 | 3.4e-07 | 1 | 0.0e+00 |
| 2-3 | 40.8725 | 4.6e-07 | 2 | 3.8e-13 |
| 4-5 | 46.4285 | 8.5e-07 | 2 | 8.0e-13 |

### Reference eigenvalues: square, E<=105
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 19.7392 | 9.4e-08 | 1 | 0.0e+00 |
| 2-3 | 49.3480 | 9.2e-07 | 2 | 1.4e-13 |
| 4 | 78.9568 | 4.5e-06 | 1 | 0.0e+00 |
| 5-6 | 98.6960 | 8.0e-06 | 2 | 2.2e-06 |

### Reference eigenvalues: triangle, E<=180
| # | E (extrapolated) | ± est. error | mult. | spread of multiplet at finest level |
|---|---|---|---|---|
| 1 | 49.3480 | 1.1e-05 | 1 | 0.0e+00 |
| 2 | 98.6960 | 1.0e-04 | 1 | 0.0e+00 |

### PINN vs FEM
| case | window centre | FEM ref (mult) | #FEM eigenvalues within 0.75 | PINN Rayleigh quotient (mean over seeds [min,max]) | rel.err | parabola on normalised loss | rel.err | parabola on recorded loss (paper-style) | rel.err | min of Lambda | time/E-point |
|---|---|---|---|---|---|---|---|---|---|---|---|
| double_well | 9.58 | 9.5764±5e-04 (1) | 1 | 9.5782 [9.5782,9.5783] | +0.019% | 9.5717 [9.5496,9.5937] | -0.05% | 9.6190 [9.5684,9.6696] | +0.44% | 0.065 | 197 s |
| double_well | 19.86 | 19.7522±3e-03 (1) | 2 | 19.7593 [19.7547,19.7640] | +0.036% | 19.7341 [19.7052,19.7630] | -0.09% | 19.6846 [19.6459,19.7232] | -0.34% | 0.077 | 231 s |
| double_well | 32.90 | 32.8735±2e-02 (2) | 3 | 32.9062 [32.8571,32.9552] | +0.099% | 32.9228 [32.8972,32.9483] | +0.15% | 32.9586 [32.9180,32.9992] | +0.26% | 0.117 | 182 s |
| double_well | 39.20 | 39.1924±1e-02 (1) | 1 | 39.1941 [39.1938,39.1944] | +0.004% | 39.2305 [39.1954,39.2655] | +0.10% | 39.1894 [39.1011,39.2777] | -0.01% | 0.108 | 174 s |
| four_well | 9.85 | 9.8542±5e-04 (1) | 1 | 9.8606 [9.8581,9.8646] | +0.066% | 9.8643 [9.8338,9.8871] | +0.10% | 9.7957 [9.7760,9.8223] | -0.59% | 0.355 | 74 s |
| four_well | 20.19 | 20.1801±3e-03 (2) | 2 | 20.1751 [20.1505,20.1878] | -0.025% | 20.3348 [20.1062,20.6807] | +0.77% | 19.9790 [19.8362,20.2386] | -1.00% | 0.478 | 65 s |
| four_well | 33.23 | 33.2104±1e-02 (2) | 2 | 33.2093 [33.2061,33.2141] | -0.003% | 33.1519 [32.9924,33.2441] | -0.18% | 33.1973 [33.0519,33.3717] | -0.04% | 0.407 | 66 s |
| four_well | 39.47 | 39.4573±1e-02 (1) | 1 | 39.4515 [39.4414,39.4578] | -0.015% | 39.2486 [38.9358,39.8181] | -0.53% | 39.3994 [39.0767,39.8865] | -0.15% | 1.054 | 71 s |
| harm_w10 | 14.28 | 14.2847±3e-07 (1) | 1 | 14.2901 [14.2899,14.2904] | +0.038% | 14.3640 [14.3351,14.3929] | +0.55% | 14.4610 [14.3547,14.5673] | +1.23% | 0.174 | 110 s |
| harm_w10 | 29.13 | 29.1349±2e-06 (2) | 1 | 29.1355 [29.1353,29.1357] | +0.002% | 29.1433 [29.1251,29.1616] | +0.03% | 29.1851 [29.1753,29.1948] | +0.17% | 0.056 | 115 s |
| harm_w10 | 45.70 | 45.0887±5e-06 (2) | 1 | 45.0892 [45.0892,45.0892] | +0.001% | 45.0959 [45.0952,45.0965] | +0.02% | 45.1012 [45.0996,45.1028] | +0.03% | 0.065 | 149 s |
| harm_w10 | 62.58 | 62.5793±2e-05 (2) | 1 | 62.5796 [62.5792,62.5800] | +0.000% | 62.6267 [62.6249,62.6285] | +0.08% | 62.6373 [62.6265,62.6480] | +0.09% | 0.076 | 195 s |
