# Exploratory: cross [-1,1]x[-a,a] U [-a,a]x[-1,1] (a=1/4) and L-shape [-1,1]^2 \ [0,1]^2  (Dirichlet, V=0)
Code: `fem_cross.py` (FEM), `cross_lib.py` (samplers, boundary factors B(x), singular enrichment, training/eval with per-epoch history and chunked MC standard errors),
`diag.py` (single-node diagnostics, hard BC via B), `pen.py` / `pen_scan.py` (weak BC), `plot_diag.py`, `plot_L.py`. Results: `experiments/results/cross_*.json`, logs in `logs/`, figures in `figures/`. Everything 1 thread, seed 0, single run (no seed statistics).

## FEM reference (P2, graded refinement at the re-entrant corners; levels listed in `results/cross_fem_*.json`)
- Cross, finest level (362k triangles): 26.1149, 48.1357 (x2), 52.2307, 59.0313, 75.3402 (x2), 90.0519 (E1..E8). Consecutive levels differ by ~1e-3 for E1, so +-1e-3; E2/E3 depend on the base mesh (48.136 vs 48.150 for a coarser base), uncertainty ~1e-2. Doubles are the expected D4 doublets (48.1357, 75.340).
- L-shape: 9.6402, 15.1984, 19.7424, 29.5307, 31.9246, 41.498, 44.968 (E1 known 9.6397, ours +5e-4).

## Boundary factors (`figures/B_cross.png`)
R-disjunction B = phi1+phi2+sqrt(phi1^2+phi2^2), phi1=(1-x^2)(a^2-y^2), phi2=(a^2-x^2)(1-y^2): zero on the boundary (max|B| = 0 on 4004 boundary points), strictly positive inside (min over 2e5 interior samples 1e-7 near the boundary; computed in a cancellation-free form), but only Lipschitz at the 4 re-entrant corners: Delta B ~ 1.9/r there (Delta B = 4, 172, 1860 at r = 0.1, 0.01, 0.001), so (Delta u)^2 ~ NN(corner)^2/r^2 is NOT integrable (log-divergent) and the Monte-Carlo residual has infinite variance. Alternatives tried: eps-smoothed sqrt (bounded Delta B but B != 0 within ~eps of the corners), B*(phi1^2+phi2^2)^{1/2} (Delta B bounded ~58, B ~ r^2), rescaled B (x0.05), and for the L the analogous construction. Plain B is also incompatible with u ~ r^{2/3} sin(2 theta/3): u/B ~ r^{-1/3} (or r^{-4/3}) is not representable by a smooth network.
I also added an exact singular enrichment u = B NN + sum_c c_c chi(r_c) r_c^{2k/3} sin(2k theta_c/3), k=1,2, with analytic Laplacian (checked against finite differences).

## Cross: diagnosis (cold start at E=26.115 = lambda_1, mu=300, 2000 epochs, 5000 pts; `figures/diag_hist_E26.1.png`, `figures/resid_*.png`)
| variant | result |
|---|---|
| baseline B; fixed sample; lr decay; 20000 pts; mu=3000; eps-smooth B | collapse u->0 (mass 0.000; training loss -> mu=300) except mu=3000: mass 0.49 but r=116 |
| enrichment (B, B x0.05, B*r with k<=2, lr 3e-3) | collapse (mass 0.000) |
| enrichment + eps-smooth B (eps=0.02) | no collapse, slow escape from the u~0 plateau (hundreds of epochs), mass 0.97, r=3.6-5 (not converged), Rayleigh quotient 23.4-23.5 < lambda_1: only because B(corner) != 0 so the BC is violated near the corners |
| weak BC (beta=100, 1000, lr decay) | no collapse, mass 0.96, r=1.4 (beta=100) but boundary violation mean_b u^2/mean u^2 = 0.29, rho_res=26.9 vs rho_var=17.7 (inconsistent): not trustworthy; beta=1000: mass 0.46, r=9.8 |
- MC variance (re-evaluation on 20 fresh chunks of 10^4 points): for the plain B the chunk-wise r varies between 35 and 218 (r = 86 +- 12 standard error), i.e. heavy-tailed, 1-2 orders of magnitude between chunks; with bounded Delta B the spread is 1-5% (e.g. r=4.98 +- 0.05). This is the "unstable, oscillating loss" of the authors: it is MC variance from the non-square-integrable residual at the corners plus optimizer jumps.
- Pointwise residual (enrichment + eps, E=26.1): 54-79% of sum(res^2) lies within 0.1 of the four corners (5.4% of the area): `figures/resid_enr_eps*.png`.
- Honest conclusion for the cross: the hard-BC route (any B tried) collapses; enrichment+smoothing stops the collapse but violates the BC; weak BC at beta=100 gives a non-collapsed but wrong (BC violated by 29%) solution; higher beta is stiff and did not converge in 2000 epochs. NOT solved. A theoretical reason that applies to all hard-BC variants: for a non-convex domain H^2 intersect H^1_0 is not dense in the graph-norm domain of the Dirichlet Laplacian (the corner function r^{2/3} sin(2theta/3) is harmonic but not H^2), so inf over smooth B*NN of ||(Delta+E)u||^2/||u||^2 stays > 0 at an eigenvalue; for the thin arms (a=1/4) the corner coefficient is large and the floor can exceed the collapse threshold 2 mu (seen: all hard-BC runs end at the u = 0 state with loss = mu). This is a hypothesis consistent with the data (not proven numerically: no direct floor computation).

## L-shape
Hard BC (`figures/L_epoch_histories.png`, E=9.64, mu=100, 2000 epochs): B (R-conjunction), B + enrichment, B*r, B*r + enrichment, plus lr decay 0.998 / lr 3e-4 / fixed sample reruns (killed when the penalty variant was requested, so only the first four finished): no collapse but NO convergence: the per-epoch residual term oscillates between 25 and 90 for 2000 epochs (r=8.4-10, mass 0.35-0.69); this reproduces the authors' "oscillating loss".
Weak BC (the authors' suggestion; plain MLP, interior 5000 pts, 1500 boundary pts uniformly on the 6 edges, mu=100 for diagnostics, beta times mean boundary u^2):
| beta, protocol | rho_res | rho_var | r | mass | boundary u^2 rel. | FEM 9.6402 |
|---|---|---|---|---|---|---|
| 100, const lr | 9.4606 | 9.464 +- 0.035 | 0.55 | 1.00 | 7e-3 | -1.9% |
| 1000, const lr | 9.5076 | 9.491 | 1.12 | 0.99 | 2.7e-3 | -1.4% |
| 100, lr decay 0.999 | 9.6055 | 9.811 | 0.68 | 0.95 | 9e-3 | -0.4% |
| 1e4, lr decay | collapse (mass 0) | | | | | |
Boundary violation (rel. to mass) decreases with beta; the eigenvalue estimate error is not monotone in beta at these training lengths (training error + boundary bias, a weak-BC problem lowers the quotient), beta=1e4 is too stiff for lr 1e-3 / 2000 epochs. Per-epoch residual histories now decay smoothly (`L_epoch_histories.png`). Pointwise residual: 33% of sum(res^2) within 0.2 of the corner (3% of the area), `figures/L_resid_pen_b100.png`.

Scan (warm start, offset grid E=6.4,7.4,...,24.4, mu=200, lr decay 0.999 per node, 2000 epochs/node, ~4 min/node/core; `figures/L_penalty_scan.png`, `results/cross_scan_L_b100_s0.json`, `..._b300_s0.json`):
| beta | detected minimum of r^2 (and of the recorded residual) | rho_res at node (rel. err) | 3-pt parabola on r^2 (rel. err) | r | boundary rel. | FEM |
|---|---|---|---|---|---|---|
| 100 | 9.4 | 9.305 (-3.5%) | 9.296 (-3.6%) | 0.36 | 4e-3 | 9.6402 |
| 100 | 15.4 | 15.334 (+0.9%) | 15.154 (-0.3%) | 0.25 | 8e-4 | 15.1984 |
| 100 | 20.4 | 20.093 (+1.8%) | 20.236 (+2.5%) | 0.46 | 1e-3 | 19.7424 |
| 300 | 9.4 | 9.440 (-2.1%) | 9.482 (-1.6%) | 0.51 | 2e-3 | 9.6402 |
| 300 | 15.4 | 15.196 (-0.02%) | 15.135 (-0.4%) | 0.33 | 2e-4 | 15.1984 |
| 300 | 19.4 | 19.595 (-0.7%) | 19.887 (+0.7%) | 0.49 | 4e-4 | 19.7424 |
All three eigenvalues in [6,25] are detected by the grid minima (E1 and E3 with an offset-grid error of 0.2-0.7 in the node, estimates 0.4-3.5% off), no eigenvalue missed, no spurious minimum. The nets between the dips are NOT eigenfunctions (rho_res plateaus, e.g. 10.2-10.8 at E=12.4 for beta=300) and the quotients drift with the node (rho_res vs rho_var differ by up to 1.5 away from eigenvalues, where the weak-BC quotient is meaningless); trust only nodes where r is small and boundary violation < 1e-3.

## What stabilizes the curve (summary)
1. Not MC noise alone but MC variance from a non-square-integrable residual: plain R-function B (Delta B ~ 1/r) gives heavy-tailed chunk spread (35-218 in r); bounded Delta B removes the spread but not the collapse.
2. lr decay and fixed samples change little for hard BC; the cross collapses for all of them.
3. On the L the decisive change was dropping B(x) and imposing the BC weakly (beta 100-300 + lr decay 0.999 per node + warm start): smooth loss curve with 3 correct minima. Same recipe on the cross (beta 100/1000) does not give a trustworthy solution within 2000 epochs.
Open: seeds, longer training / beta continuation for the cross, direct estimate of the H^2-class floor.
