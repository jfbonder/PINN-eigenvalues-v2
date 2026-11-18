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
5. **Close eigenvalues, harmonic w=10 (corrected in round 2, see section 8)**: the window around 45.7 contains the double eigenvalue 45.0887 and the simple 46.2907. The warm-started chain stays on the 45.09 branch up to E=46.2-46.7 (loss peak, hysteresis) and then switches: the Rayleigh quotient is 46.289-46.292 at the last nodes (seed 0: 47.2, seed 1: 46.7). So 46.29 is found, but only after the node chain passes ~46.2-46.7; with the paper's coarse step (>=0.7) the dip is easily missed.
6. Cost (1 core, this container, load from other jobs, so +-50%): PINN 197 s (double well), 69 s (four well), 142 s (w=10 disk) per E-point on average (early stopping: 840-1050 epochs on average); a 7-point window 7-25 min, a paper-style 120-point sweep of [0,75] would take ~7 h (double well). FEM: <10 s in 2-D (32k dof) and 112-134 s for the 3-D ball (42k dof, 60 eigenpairs, all eigenvalues in [0,75] with multiplicities, error 1e-3..1e-1).
   FEM is therefore 2-3 orders of magnitude cheaper and more accurate for these linear problems (as the paper itself says: the method is not meant to compete with FEM on linear problems).

## 6. Discrepancies with the paper (v3)
- **D1 harmonic potential**: the paper text/code `V_harmonic` use V=(w^2/2)|x|^2; the original notebook code used w^2|x|^2. FEM: (w^2/2): 5.8919 (w=1), 14.2847 (w=10); (w^2): 6.0000 (w=1), 20.0161 (w=10). The figures' minima (w=10: ~14.3, ~29, ~45, ~62) and the PINN values (14.29) match the w^2/2 version; the paper's text is right and the original notebook code is not what produced the figures. (The values 5.8931/14.2944 quoted in the brief are within 0.02%/0.07% of FEM 5.8919/14.2847.)
- **D2 annulus (Fig. 9)**: confirmed: eigenvalues in [0,50] are 39.0133 (1), 40.8725 (2), 46.4285 (2): five eigenfunctions, three distinct values; the paper's theoretical curve has one zero and the PINN curve misses two.
- **D3 harmonic w=10 (Fig. 12)**: 45.0887 (x2) and 46.2907 are separated by 1.2; a step-0.5 chain resolves both (section 8), a coarse step with warm-start hysteresis does not.
- **D4 wells**: double well centres sit on the boundary; E1=9.5764 vs 9.8696 free; four well E1=9.8542, practically free ball. No eigenvalues are reported in the paper for these cases; the table above can be added. The paper's loss curves (Figs. 17-18) show single minima at ~9.6, 19.9, 33, 39.3, 48.6, 59.5, 66.7: consistent with FEM cluster positions but each of the later minima stands for 3-9 eigenvalues.
- Four-well depth is e^{-1}=0.37 per well, not 1 (V0 multiplies phi whose max is e^{-1}).

## 7. Limitations
3-D FEM has only 3 mesh levels with observed order ~2.8, error estimates are of Richardson type (verified conservative on the free ball). PINN windows were centred with FEM knowledge and 2-3 seeds only; the extra seeds of the harmonic and double-well cases were cut for time (container restart). Model weights for runs after the first double-well seed are in `experiments/results/fem_pinn_models/`.

## 8. Round 2 - summary (referee H, items 4, 5, 10, 15; all 1 core, seed 0 unless stated; scripts `pinn_round2.py`, `mc_eval.py`, `make_round2.py`, logs `logs/round2_*.log`, results `experiments/results/fem_r2_*_s0.json`, `fem_mc_se.json`, LaTeX in `tables_round2.tex`)
- **Item 4.** Per-seed errors of the closed-form Rayleigh quotient: worst single run +0.248% (double well 32.9 window, which contains three FEM groups spanning 32.79-33.03, so the value lies inside the cluster); next worst -0.147% and +0.105% (four well); isolated windows <= 0.06%; harmonic w=10 <= 0.04%. So: seed means within 0.1%, single runs within 0.25%. FEM uncertainty column added (Richardson estimates 5e-4 ... 2e-2; they exceed the PINN error in several 3-D windows, e.g. 9.5764 ± 5e-4 vs PINN 9.5782). Blind scan (double well, nodes 4.7:1.5:49.7, mu=400, i.e. not aligned with the spectrum): the paper-style loss curve has 5 minima (9.2, 19.7, 33.2, 39.2, 48.2) for 17 eigenfunctions / 10 distinct FEM eigenvalues in [0,49.7]; every minimum is within the FEM cluster it stands for (parabola on the recorded loss is off by up to 0.5 at the clusters, e.g. 32.70 vs 32.79-33.03 and 48.55 vs 48.41-48.68, because the recorded loss is not a parabola in the mu=400 regime; the Rayleigh quotient of the net at the node is within 0.04% of a cluster member). Multiplicities/cluster members are not visible in the curve.
- **Item 5 (reviewer is right).** In the existing windows the last nodes (seed 0: E=47.2, seed 1: E=46.7) have Rayleigh quotient 46.289-46.291: the network DOES switch to the 46.29 branch; my earlier statement "not found" only reflected the window end. Extended window (seed 0, E=45.2..49.2, step 0.5): rho=45.093 at E=45.2/45.7, and 46.290 (+-0.0004) from E=46.2 on (up to 49.2); the normalised loss has a separate minimum at E=46.2 (r^2=0.06 vs 0.45 at 45.7 and 0.19 at 46.7, i.e. a dip in front of the 46.29 level separated from the 45.09 dip by a ridge at 45.7), so with step 0.5 both eigenvalues are visible. What remains true: with a coarse step (>= 0.7, the paper) and warm start, the hysteresis (stays on 45.09 up to ~46.2-46.7) hides it. Parabola/recorded-loss detection must use the node chain, not the Rayleigh quotient, to state this.
- **Item 10 (Tables 2-4 re-run with local offset windows, nodes exact-0.9/+0.1/+1.1, paper mu/settings, tables in 8.4).** 3-D ball (Table 2) E1..E4: rho = 9.8757 (+0.062%), 20.1912 (+0.002%), 33.2178 (+0.001%), 39.4789 (+0.001%) - as good as or better than the paper's 0.04-0.16%, r = 0.22-0.41; parabola fits are worse (up to 0.12%). 4-D ball (Table 3): 14.7122 (+0.21%), 26.3980 (+0.09%), comparable to the paper's 0.27%/0.45% - in 4-D the quotient follows E (14.507, 14.712, 14.963 at E=13.78, 14.78, 15.78), i.e. the network is not converged to an eigenfunction with 2000 epochs and the quotient is only reliable at the min-loss node. Triangle (Table 4, mu=15000): 49.3509 (+0.006%), 98.7067 (+0.011%), 128.4612 (+0.12%), 167.7923 (+0.005%) but with r = 0.9, 1.8, 5.7, 2.5 (paper: 0.49-1.47): short cold-started windows do not reach the paper's residual level, and the **first (cold) node of every triangle window failed** (rho 56.0, 57.1, 88.0, 164.7 with r = 15-62, i.e. wrong branch/unconverged after <= 2000 epochs at mu=15000), while the first nodes of the 3-D windows converged already from the cold node (r 0.9-1.0 there, quotient within 0.01 of the eigenvalue) and those of the 4-D windows only partly (rho 14.51 and 26.16). Hence targeted local windows work only after one or two warm-started nodes; this supports the referee's item-1 reading (a fresh network at a chosen E is not reliable for higher eigenvalues). Only the first 4 / 2 windows of Tables 2 / 3 were run (budget); the disk is not among Tables 2-4 of v4.
- **Item 15 (MC standard errors, 4e5 points in 40 chunks).** The closed-form quotient rho_res = -int u(Lu-Vu)/int u^2 has SE 2e-4 ... 2.4e-3 (nearly zero-variance when u is close to an eigenfunction), r has SE 1e-3 ... 1.5e-2 relative 0.1-0.4%, mass SE 0.2-0.4%. The variational quotient (int|grad u|^2+Vu^2)/int u^2 has SE 0.02-0.14, i.e. 50-300x larger: this explains the 0.1-0.6 discrepancy between rho_var and rho_res seen earlier (the 1e5-point values of double-well seed 0 are also affected); rho_res should be reported, rho_var must not be. SE columns are in every 8.3-8.5 table (`rho_res_se`, `r_se` fields in the JSON). These SE are statistical (fixed network); the dispersion between seeds (8.1) is the larger error.

### 8.1 Per-seed worst case and FEM uncertainty (item 4)
| example | window | FEM ± est.err (mult) | other FEM eigenvalues within 0.75 | rho per seed | rel.err per seed (%) | worst |
|---|---|---|---|---|---|---|
| double_well | 8.1-11.1 | 9.5764 ± 5e-04 (1) | - | 9.5783, 9.5782 | +0.019, +0.018 | 0.019 |
| double_well | 18.4-21.4 | 19.7522 ± 3e-03 (1) | 19.967(x2) | 19.7640, 19.7547 | +0.060, +0.013 | 0.060 |
| double_well | 31.4-34.4 | 32.8735 ± 2e-02 (2) | 32.786(x1), 33.034(x2) | 32.9552, 32.8571 | +0.248, -0.050 | 0.248 |
| double_well | 37.7-40.7 | 39.1924 ± 1e-02 (1) | - | 39.1944, 39.1938 | +0.005, +0.003 | 0.005 |
| four_well | 8.3-11.3 | 9.8542 ± 5e-04 (1) | - | 9.8593, 9.8646, 9.8581 | +0.052, +0.105, +0.040 | 0.105 |
| four_well | 18.7-21.7 | 20.1801 ± 3e-03 (2) | 20.189(x1) | 20.1505, 20.1878, 20.1870 | -0.147, +0.038, +0.034 | 0.147 |
| four_well | 31.7-34.7 | 33.2104 ± 1e-02 (2) | 33.214(x3) | 33.2141, 33.2077, 33.2061 | +0.011, -0.008, -0.013 | 0.013 |
| four_well | 38.0-41.0 | 39.4573 ± 1e-02 (1) | - | 39.4578, 39.4414, 39.4554 | +0.001, -0.040, -0.005 | 0.040 |
| harm_w10 | 12.8-15.8 | 14.2847 ± 3e-07 (1) | - | 14.2899, 14.2904 | +0.036, +0.040 | 0.040 |
| harm_w10 | 27.6-30.6 | 29.1349 ± 2e-06 (2) | - | 29.1357, 29.1353 | +0.003, +0.001 | 0.003 |
| harm_w10 | 44.2-47.2 | 45.0887 ± 5e-06 (2) | - | 45.0892, 45.0892 | +0.001, +0.001 | 0.001 |
| harm_w10 | 61.1-64.1 | 62.5793 ± 2e-05 (2) | - | 62.5792, 62.5800 | -0.000, +0.001 | 0.001 |

Worst single-run error of the Rayleigh quotient over all windows/seeds: 0.248%. (Seed means are within 0.1%, single runs within 0.25%; the 0.25% case is double well 32.9, a window containing 3 FEM eigenvalues with spread 0.25, i.e. inside the cluster.) The FEM uncertainty (Richardson estimate, conservative) is 5e-4..2e-2 in 3-D and exceeds some of the PINN errors; for clusters the reference is the cluster, not a single member.

### 8.2 Blind scan of the double-well loss curve vs FEM (item 4)
Blind scan, double well, nodes E=4.7..49.7 step 1.5 (31 nodes done), mu=400, 1 seed. Local minima of the recorded loss (paper-style detection):

| min at node E | recorded loss | normalised loss r^2 | parab. fit (recorded) | Rayleigh quotient at node | FEM eigenvalues within 1.5 (mult) | nearest FEM | err of rho |
|---|---|---|---|---|---|---|---|
| 9.2 | 0.185 | 0.187 | 9.576 | 9.5779 | 9.576(x1) | 9.5764 | +0.015% |
| 19.7 | 0.306 | 0.163 | 19.767 | 19.7545 | 19.752(x1), 19.967(x2) | 19.7522 | +0.012% |
| 33.2 | 0.553 | 0.524 | 32.698 | 32.7966 | 32.786(x1), 32.874(x2), 33.034(x2) | 32.7856 | +0.034% |
| 39.2 | 0.127 | 0.126 | 39.222 | 39.1945 | 39.192(x1) | 39.1924 | +0.005% |
| 48.2 | 0.386 | 0.421 | 48.554 | 48.5270 | 48.406(x1), 48.430(x2), 48.545(x2), 48.676(x2) | 48.5454 | -0.038% |

FEM eigenvalue groups (E<52) with NO detected minimum within 0.75 (distinct eigenvalues listed individually; clusters = several within a few tenths): none

Counting: FEM has 11 distinct eigenvalue groups / 17 eigenfunctions in [0,49.7]; the scan shows 5 separate minima (9.58, 19.75/19.97, 32.79/32.87/33.03, 39.19, 48.41-48.68). The Rayleigh quotient of the network at the node reached the individual members only occasionally (e.g. 19.968 at E=22.7, 32.797 at E=33.2).

Node values: 4.7: loss 21.45, rho 9.579; 6.2: loss 11.58, rho 9.579; 7.7: loss 3.68, rho 9.579; 9.2: loss 0.18, rho 9.578; 10.7: loss 1.34, rho 9.578; 12.2: loss 6.91, rho 9.578; 13.7: loss 16.29, rho 9.578; 15.2: loss 20.90, rho 19.723; 16.7: loss 9.05, rho 19.753; 18.2: loss 2.70, rho 19.757; 19.7: loss 0.31, rho 19.754; 21.2: loss 2.31, rho 19.764; 22.7: loss 7.61, rho 19.968; 24.2: loss 17.85, rho 19.969; 25.7: loss 34.30, rho 19.967; 27.2: loss 31.46, rho 32.753; 28.7: loss 16.69, rho 32.789; 30.2: loss 6.94, rho 32.791; 31.7: loss 1.26, rho 32.791; 33.2: loss 0.55, rho 32.797; 34.7: loss 4.14, rho 32.800; 36.2: loss 10.15, rho 33.024; 37.7: loss 2.62, rho 39.195; 39.2: loss 0.13, rho 39.194; 40.7: loss 2.48, rho 39.192; 42.2: loss 8.94, rho 39.196; 43.7: loss 20.01, rho 39.195; 45.2: loss 34.57, rho 39.193; 46.7: loss 3.84, rho 48.521; 48.2: loss 0.39, rho 48.527; 49.7: loss 1.62, rho 48.531

### 8.3 Harmonic w=10, window extended to 49.2 (item 5)
| E | recorded loss | normalised r^2 | rho (net) ± SE | r | mass |
|---|---|---|---|---|---|
| 45.2 | 0.097 | 0.097 | 45.0930 ± 0.0004 | 0.312 | 0.971 |
| 45.7 | 0.439 | 0.446 | 45.0952 ± 0.0005 | 0.668 | 1.013 |
| 46.2 | 0.072 | 0.058 | 46.2903 ± 0.0004 | 0.240 | 1.016 |
| 46.7 | 0.196 | 0.191 | 46.2905 ± 0.0002 | 0.437 | 1.007 |
| 47.2 | 0.880 | 0.890 | 46.2919 ± 0.0004 | 0.943 | 1.002 |
| 47.7 | 1.931 | 2.086 | 46.2894 ± 0.0006 | 1.444 | 0.986 |
| 48.2 | 3.638 | 3.701 | 46.2912 ± 0.0004 | 1.924 | 0.977 |
| 48.7 | 5.693 | 5.932 | 46.2891 ± 0.0004 | 2.435 | 0.961 |
| 49.2 | 8.298 | 8.540 | 46.2904 ± 0.0004 | 2.922 | 0.962 |

### 8.4 Tables 2-4 re-run with local offset windows (item 10)
**Table 4 (triangle, mu=15000)**: 3 offset nodes (exact-0.9, +0.1, +1.1) per eigenvalue, fresh network at the first node, warm start along the window; rho = closed-form frozen-network Rayleigh quotient (4e5 MC pts, standard error in brackets).

| exact | rho (min-loss node) ± SE | rel.err | parabola on r^2 | rel.err | r ± SE | mass | min r^2 | paper value (v4) | paper rel.err |
|---|---|---|---|---|---|---|---|---|---|
| 49.3480 | 49.3509 (0.0014) | +0.0058% | 49.947 | +1.213% | 0.945 ± 0.002 | 0.997 | 0.893 | 49.3699 | +0.0444% |
| 98.6960 | 98.7067 (0.0024) | +0.0108% | nan | +nan% | 1.811 ± 0.003 | 1.011 | 3.279 | 98.8922 | +0.1987% |
| 128.3049 | 128.4612 (0.0095) | +0.1219% | nan | +nan% | 5.676 ± 0.015 | 0.986 | 32.214 | 128.0485 | -0.1998% |
| 167.7833 | 167.7923 (0.0035) | +0.0054% | nan | +nan% | 2.513 ± 0.003 | 1.006 | 6.314 | 167.6872 | -0.0573% |

All nodes (E: Rayleigh quotient, r, epochs; a node with r >> 1 or rho far from E has NOT converged to the targeted eigenfunction - cold start at the first node of each window):

- window 49.3480: 48.45: rho=55.965, r=44.20, ep=874; 49.45: rho=49.351, r=0.94, ep=1506; 50.45: rho=49.358, r=1.83, ep=925
- window 98.6960: 97.80: rho=57.121, r=47.41, ep=2000; 98.80: rho=99.044, r=4.46, ep=2000; 99.80: rho=98.707, r=1.81, ep=1655
- window 128.3049: 127.40: rho=88.028, r=61.92, ep=2000; 128.40: rho=135.850, r=20.51, ep=2000; 129.40: rho=128.461, r=5.68, ep=2000
- window 167.7833: 166.88: rho=164.660, r=15.01, ep=2000; 167.88: rho=167.735, r=3.37, ep=2000; 168.88: rho=167.792, r=2.51, ep=1793

**Table 2 (3-D ball, mu=100)**: 3 offset nodes (exact-0.9, +0.1, +1.1) per eigenvalue, fresh network at the first node, warm start along the window; rho = closed-form frozen-network Rayleigh quotient (4e5 MC pts, standard error in brackets).

| exact | rho (min-loss node) ± SE | rel.err | parabola on r^2 | rel.err | r ± SE | mass | min r^2 | paper value (v4) | paper rel.err |
|---|---|---|---|---|---|---|---|---|---|
| 9.8696 | 9.8757 (0.0004) | +0.0616% | 9.857 | -0.125% | 0.409 ± 0.001 | 0.954 | 0.167 | 9.8850 | +0.1562% |
| 20.1907 | 20.1912 (0.0004) | +0.0022% | 20.184 | -0.035% | 0.216 ± 0.000 | 0.951 | 0.047 | 20.1826 | -0.0404% |
| 33.2175 | 33.2178 (0.0004) | +0.0009% | 33.222 | +0.013% | 0.261 ± 0.000 | 0.981 | 0.068 | 33.2434 | +0.0781% |
| 39.4784 | 39.4789 (0.0006) | +0.0013% | 39.527 | +0.123% | 0.240 ± 0.001 | 1.021 | 0.058 | 39.4942 | +0.0399% |

All nodes (E: Rayleigh quotient, r, epochs; a node with r >> 1 or rho far from E has NOT converged to the targeted eigenfunction - cold start at the first node of each window):

- window 9.8696: 8.97: rho=9.870, r=0.92, ep=1689; 9.97: rho=9.876, r=0.41, ep=1727; 10.97: rho=9.870, r=1.11, ep=564
- window 20.1907: 19.29: rho=20.191, r=0.92, ep=1927; 20.29: rho=20.191, r=0.22, ep=514; 21.29: rho=20.192, r=1.14, ep=579
- window 33.2175: 32.32: rho=33.217, r=0.94, ep=1649; 33.32: rho=33.218, r=0.26, ep=506; 34.32: rho=33.218, r=1.13, ep=412
- window 39.4784: 38.58: rho=39.475, r=1.02, ep=2000; 39.58: rho=39.479, r=0.24, ep=1411; 40.58: rho=39.479, r=1.12, ep=501

**Table 3 (4-D ball, mu=200)**: 3 offset nodes (exact-0.9, +0.1, +1.1) per eigenvalue, fresh network at the first node, warm start along the window; rho = closed-form frozen-network Rayleigh quotient (4e5 MC pts, standard error in brackets).

| exact | rho (min-loss node) ± SE | rel.err | parabola on r^2 | rel.err | r ± SE | mass | min r^2 | paper value (v4) | paper rel.err |
|---|---|---|---|---|---|---|---|---|---|
| 14.6820 | 14.7122 (0.0005) | +0.2060% | 14.631 | -0.346% | 0.455 ± 0.001 | 1.074 | 0.207 | 14.6418 | -0.2736% |
| 26.3746 | 26.3980 (0.0010) | +0.0886% | 26.242 | -0.503% | 0.791 ± 0.001 | 1.065 | 0.626 | 26.2571 | -0.4457% |

All nodes (E: Rayleigh quotient, r, epochs; a node with r >> 1 or rho far from E has NOT converged to the targeted eigenfunction - cold start at the first node of each window):

- window 14.6820: 13.78: rho=14.507, r=0.82, ep=1772; 14.78: rho=14.712, r=0.46, ep=738; 15.78: rho=14.963, r=1.03, ep=474
- window 26.3746: 25.47: rho=26.156, r=0.90, ep=1913; 26.47: rho=26.398, r=0.79, ep=875; 27.47: rho=26.659, r=1.06, ep=456

### 8.5 Monte Carlo standard errors (item 15)
| run (saved network) | rho_res ± SE | rho_var ± SE | r ± SE | mean u^2 ± SE |
|---|---|---|---|---|
| harm_w10_s0_E45.200 | 45.0889 ± 0.0002 | 45.199 ± 0.068 | 0.233 ± 0.001 | 0.994 ± 0.002 |
| harm_w10_s0_E14.280 | 14.2882 ± 0.0006 | 14.314 ± 0.028 | 0.392 ± 0.001 | 0.936 ± 0.002 |
| four_well_s0_E9.850 | 9.8580 ± 0.0007 | 9.860 ± 0.022 | 0.577 ± 0.001 | 1.012 ± 0.002 |
| four_well_s0_E39.470 | 39.4519 ± 0.0024 | 39.447 ± 0.144 | 1.507 ± 0.003 | 1.014 ± 0.004 |
| double_well_s1_E19.860 | 19.7551 ± 0.0005 | 19.830 ± 0.045 | 0.272 ± 0.000 | 1.026 ± 0.002 |
| double_well_s1_E39.200 | 39.1942 ± 0.0005 | 39.192 ± 0.141 | 0.390 ± 0.001 | 1.003 ± 0.004 |

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
