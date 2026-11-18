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
