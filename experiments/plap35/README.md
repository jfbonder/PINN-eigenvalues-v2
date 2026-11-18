# p-Laplacian with p = 3.5 on the unit disk — exploratory study (not part of the paper revision)

Question: the strong-form method degrades for large p (flat curve at p = 4; the authors recall that for p > 3 "everything became very
unstable and the loss curve oscillated"). What exactly goes wrong at p = 3.5, and which change fixes it?

## Files
`refs35.py` (references) · `core35.py` (all loss variants, per-epoch histories, MC-spread diagnostics, resumable) · `jobs35.py`, `run35.py`
(queues A–D) · `analyze35.py` → `tables35.md`, `figures/p35_curves.png`, `p35_fine_curves.png`, `p35_training_histories.png`,
`p35_residual_maps.png`, `results/plap35_summary.json` · raw records `results/plap35_<job>.json` · `logs/`.
`pinn_eigen.py` and `paper/` are untouched.

## Settings
3×128 tanh × (1−|x|²); N = 5000 fresh points per epoch; Adam; μ = 200; P1 = lr 1e-3 → 1e-5 cosine over 1500 epochs per node, warm start
along the grid; P0 = constant lr 1e-3, ≤ 2000 epochs, early stopping (patience 300, tol 1e-5). Main window: 5 nodes E = 10.5 … 14.5
(h = 1, not centred on the reference); fine scans: 13 nodes 11 … 14 (h = 0.25). Seeds 0, 1, 2 where stated, otherwise seed 0.
Every loss value below is evaluated on a fresh sample of 40 000 points; the same sample split into 8 batches of 5000 gives the
"MC std" (spread of the loss of one *frozen* network over batches of the training size). "E_PINN" = closed-form minimiser in E of the
variant's functional for the frozen network; "self-consistent E*" = the E at which E_PINN(E) = E (linear interpolation between nodes).
Training time: 9.6 core-hours in total (two processes, ≈ 5.1 h wall, machine shared); per node ≈ 3.3 min (strong form), ≈ 1.7 min (mixed).

## References
E_1 = 12.15465 (radial shooting, stable to 3·10⁻¹⁰; P1-FEM + Richardson: 12.15464). Half disk (one nodal diameter; upper bound for the
second eigenvalue, equal to it if the second eigenfunction is antisymmetric): 69.160. Quarter disk: 180.88. Radial nodal-circle mode: 280.35.

## Findings

**1. The baseline is not "unstable" in the sense of diverging; it is flat and noisy, and the noise is Monte Carlo noise.**
* 5-node curve (3 seeds, P1): 9.8/5.4/9.9, 2.4/2.2/3.6, **2.05/1.67/2.17**, 2.8/2.9/2.9, 5.6/6.3/5.8 — the three central nodes differ by
  less than the seed spread. 79–82 % of the loss at E = 12.5 comes from the 2 % of points with the smallest |∇u| (residual map: one bright
  spot at the maximum of u). 3-point vertex: +1.5, +1.3, **+4.2 %**. No collapse in the E_1 window.
* **Sampling variance is the source of the oscillation.** For one frozen baseline network at E = 12.5 the loss on batches of 5000 points is
  2.05 ± 0.53 (range 1.5–3.0 over 8 batches; ± 0.36 and ± 0.96 for the other two seeds), i.e. ±25–45 %. The training-batch residual over
  the last 300 epochs fluctuates by the same amount with lr decay (1.65 ± 0.52), with constant lr (1.43 ± 0.52) and with P0 (1.65 ± 0.56):
  **the learning-rate schedule makes no difference at p = 3.5**, unlike at p = 2.2 where optimizer noise dominated. The loss rises by only
  ≈ 0.4–0.8 between neighbouring nodes at h = 1 and by ≈ 0.1 at h = 0.25, so any curve built from last-batch values (as in the original
  notebooks) or from h ≤ 0.5 oscillates. Fine scan, P0: 1.08, 1.33, 1.30, 0.92, 1.36 at E = 12.0 … 13.0 (argmin 12.75, **+4.9 %**); fine
  scan, P1: 0.67, 0.59, 0.71, 0.67 at 12.0 … 12.75 (flat within the MC error ≈ 0.37/√8 of the 40 000-point evaluation).
* The floor itself is mainly an optimisation/approximation effect of the defect at the critical point and falls slowly with training:
  2.05 (1500 epochs at the node), 0.68 (5000 epochs), 0.71 (fine scan, i.e. several previous nodes nearby) — but the MC std does not
  fall in proportion (0.26 … 0.37), so the curve stays noisy.
* The refined value is not robust either: E_PINN at node 12.5 is 12.108, 12.147, 12.112 (−0.38, −0.06, −0.35 %), but it moves with the
  node, 11.93–12.35 over nodes 11.5–13.5 (−1.9 … +1.6 %), about 0.17 per unit of E. The self-consistent value is 12.064, 12.075, 12.024
  (−0.75, −0.65, −1.07 %).

**2. Remedies requested — outcome.**
* **(a) Mixed formulation** (network outputs u and σ; ∫(div σ + E|u|^{p−2}u)² + λ∫|σ − |∇u|^{p−2}∇u|² + μ(…)²).
  It **strongly reduces the sampling noise**: for a frozen network at E = 12.5 (λ = 100, seed 0) the total loss on batches of 5000 is
  1.64 ± 0.13 (8 %, range 1.42–1.78; baseline 26–45 %); the residual term alone is 0.36 ± 0.007, so what noise remains sits in the
  constraint term (the "MC std" column of Table 1 is the residual term only for the mixed rows; for λ = 10 the total was not measured,
  but the 40 000-point values of three seeds agree within 0.05). The fine curve is smooth and convex, and an epoch costs half as much
  (no second derivatives of u). It does **not** remove
  the defect: the pointwise residual just moves into the constraint term, a ring around the maximum of u (residual map), where the smooth
  u cannot reproduce the flux; with λ = 100 the constraint carries ≈ 80 % of the loss and the curve is as flat as the baseline
  (vertex +1.4, +2.0, +3.5 %; fine-scan vertex +0.8 %). **λ = 10 gives the best curve of the requested variants**: 1.95, 0.56, 0.36,
  1.30, 3.34 (three seeds agree to 0.05), vertex 12.171, 12.161, 12.181 (**+0.14, +0.06, +0.22 %**). Its closed-form E_PINN is poor
  (+1.2, +0.8, +1.3 % at node 12.5, 11.86–12.79 over the central nodes) because with a weak constraint div σ follows E rather than u;
  the self-consistent value is −0.08, −0.51, −0.03 %. λ = 1000 fails (collapse to u ≡ 0 at three nodes, never recovers a proper curve).
  The first node from scratch collapsed in 2 of 5 mixed runs with λ = 100 (seed 0 with P1 and with P0) and recovered at the next node.
* **(b) Weighting.** Down-weighting the residual by ω = |∇u|²/(|∇u|²+κ²), κ = 0.3 (this minimises a weighted L² residual whose weight
  depends on u; its zeros are still exactly the eigenpairs): loss 1.20 instead of 2.05, MC std 0.32, core fraction still 0.77, vertex
  +2.9 %, E_PINN +0.27 % (12.09–12.39 over nodes) — **no real change** (one seed). Dividing the residual by (|∇u|²+κ²)^{(p−2)/2}, as
  literally suggested, **makes it worse** (no minimum, loss 12–21, MC std 10.7, E_PINN −15 %): for p > 2 the division amplifies exactly
  the points near ∇u = 0 (it is the "normalized p-Laplacian" form, in which the right-hand side E|u|^{p−2}u/|∇u|^{p−2} is unbounded
  at the critical point even for the exact eigenfunction).
* **(c) More points / importance sampling.** N = 20 000: loss 2.24, MC std 0.42 (per 5000), vertex +5.9 %, E_PINN −0.37 % — **no
  improvement** at 4× the cost. Importance sampling (half of the points uniformly in r < 0.3, exact weights; this uses the knowledge that
  the critical point is at the origin): loss 1.39, core fraction 0.83, vertex +1.2 %, E_PINN −0.19 % — marginal.
* **(d) Larger μ / longer training.** μ = 1000: identical to μ = 200 (2.21, vertex +1.8 %). 5000 epochs: floor 0.68 instead of 2.05, but
  vertex +3.2 % and E_PINN +0.26 %, node range unchanged — lower floor, same flat bottom.

**3. An additional remedy that addresses the cause: a singular input feature (e).**
The exact eigenfunction behaves like u(x₀) − c|x−x₀|^{p/(p−1)} at a nondegenerate extremum (exponent 1.4 for p = 3.5), which no smooth
network can represent. Giving the network the extra input ρ = |x − c|^{p/(p−1)} (c = 0, or c trainable — identical results, the trained
centre stays at the origin) with the *unchanged* strong-form loss gives:
* floor 0.22, 0.15, 0.21 at node 12.5 (10× lower than baseline) and 0.055 in the fine scan (≈ 11× lower than the baseline fine scan,
  ≈ 40× lower than the baseline h = 1 value); core fraction 0.3; MC std 0.012–0.036 (h = 1) and 0.005 (fine);
* a sharp curve: 5.2, 0.64, 0.22, 1.67, 4.87; fine scan 0.48, 0.21, 0.080, 0.055, 0.16, 0.42 with a single node within 25 % of the minimum
  (baseline: four nodes, 12.0–12.75);
* curve vertex +0.57, +0.06, +0.79 % (h = 1, three seeds), **+0.16 %** (fine scan, 3 points; −0.01 % with a 5-point parabola);
* E_PINN at node 12.5: +0.71, +0.45, +0.27 % and still node-dependent (12.01–12.44); self-consistent value +0.19, +0.07, −0.29 %
  (fine scan +0.30 %).
So the feature fixes the *curve* (detection and location of the minimum) but not the node dependence of the closed-form refinement.
Caveat: it needs the location of the critical points (trivial for a first eigenfunction with a symmetry centre; for sign-changing modes
one would need one trainable centre per extremum — not tested).

**4. Second eigenvalue (reference 69.16): not obtained.** Started from scratch at E = 65 … 73 (3000 epochs at the first node), the
baseline stays essentially in the trivial state (mass = ∫|u|^p = 0.007–0.06 at four nodes, loss 880–2300); a 500-epoch warm-up on the
normalisation term alone does not help (mass 0.006–0.23). Only the last node E = 73 reaches mass ≈ 0.5, with E_PINN = 69.31 and 69.39
(+0.2, +0.3 % of the half-disk value) but a loss of ≈ 180–190, so this is an indication, not a result. The mixed formulation collapses
at all five nodes (mass ≈ 0.004). For p > 2 the penalty μ(∫|u|^p − 1)² has a vanishing gradient at u = 0 (∝ |u|^{p−1}), and at large E the
residual term pushes towards u = 0 faster than the penalty pulls away; a continuation in E from the first eigenvalue, or a different
normalisation/initial scaling, would be needed. We did not have the budget to try them.

## Honest diagnosis
* What destabilises the curve for p > 3 is the **heavy-tailed pointwise residual at the critical point**: the loss of a fixed network
  fluctuates by 25–45 % between batches of 5000, more than the loss varies between neighbouring nodes. Learning-rate decay, larger μ,
  more uniform points and longer training do not change this.
* What stabilises the curve: **the mixed formulation** (batch-to-batch spread 8 % instead of 26–45 % at λ = 100; with λ = 10 the vertex is within +0.06 … +0.22 % for three
  seeds) and **the singular feature** (floor ÷ 10–40, sharp minimum, vertex +0.06 … +0.8 % at h = 1 and +0.16 % at h = 0.25).
* What nobody fixes: the closed-form refinement drifts with the training node by ≈ 0.17 per unit of E in every variant, so at p = 3.5 the
  value should be read from the (stabilised) curve or from the self-consistent point, not from a single node. Achieved accuracy for E_1
  is ≈ 0.1–0.3 % with the two working variants, against 1.3–4 % (curve) for the baseline. The baseline E_PINN at the node nearest to
  the eigenvalue is also within 0.4 %, but only if that node is already known.
* Not done: more than one seed for (b), (c), (d) and the fine scans; combining the mixed formulation with the singular feature (the
  natural next step: the feature removes the constraint defect, the mixed form removes the second derivatives); any λ between 10 and 100;
  second eigenvalue with continuation.

## Tables (generated by `analyze35.py`; errors in % of E_1 = 12.15465)

References: E1 = 12.15465 (radial shooting; P1-FEM 12.15464); half disk (1 nodal diameter) 69.1602; quarter disk 180.878; radial 2nd 280.35.

### Table 1: 5-node window (E = 10.5 … 14.5, h = 1; N20000/ep5000: 3 nodes 11.5–13.5).  loss = variant functional on a fresh sample of 40000

| variant | loss at nodes | grid min | 3-pt vertex (err %) | E_PINN at node 12.5 (err %) | E_PINN over nodes 11.5–13.5 (err %) | core frac @12.5 | MC std of loss @12.5 (batches of 5000) | collapsed nodes | wall min | self-consistent E* (err %) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (strong form), P1, seed 0 | 9.78, 2.42, 2.05, 2.80, 5.58 | 12.5 | 12.331 (+1.45) | 12.108 (-0.38) | 12.01–12.35 (-1.22 to +1.65) | 0.81 | 0.532 | – | 18 | 12.064 (-0.75) |
| baseline, seed 1 | 5.43, 2.22, 1.67, 2.87, 6.25 | 12.5 | 12.316 (+1.33) | 12.147 (-0.06) | 11.98–12.35 (-1.45 to +1.59) | 0.79 | 0.362 | – | 16 | 12.075 (-0.65) |
| baseline, seed 2 | 9.94, 3.65, 2.17, 2.94, 5.77 | 12.5 | 12.659 (+4.15) | 12.112 (-0.35) | 11.93–12.31 (-1.86 to +1.24) | 0.82 | 0.956 | – | 18 | 12.024 (-1.07) |
| baseline, P0 (const lr + early stop), seed 0 | 5.65, 2.51, 1.50, 3.35, 6.66 | 12.5 | 12.352 (+1.63) | 12.191 (+0.30) | 12.06–12.30 (-0.77 to +1.16) | 0.67 | 0.588 | – | 15 | 12.145 (-0.08) |
| baseline, P0, seed 1 | 5.09, 1.80, 1.23, 2.67, 6.00 | 12.5 | 12.282 (+1.05) | 12.180 (+0.21) | 12.01–12.33 (-1.15 to +1.41) | 0.79 | 0.477 | – | 11 | 12.117 (-0.31) |
| baseline, P0, seed 2 | 5.66, 2.70, 1.61, 3.21, 6.64 | 12.5 | 12.405 (+2.06) | 12.145 (-0.08) | 11.97–12.35 (-1.48 to +1.58) | 0.71 | 0.491 | – | 13 | 12.072 (-0.68) |
| baseline, constant lr, 1500 epochs | 6.48, 1.92, 1.75, 2.80, 5.42 | 12.5 | 12.141 (-0.11) | 12.102 (-0.43) | 11.99–12.41 (-1.37 to +2.13) | 0.73 | 0.522 | – | 17 | 12.051 (-0.86) |
| (c) N = 20000 | 8.04, 2.24, 3.06 | 12.5 | 12.875 (+5.93) | 12.110 (-0.37) | 11.79–12.32 (-3.00 to +1.40) | 0.81 | 0.424 | – | 43 | 11.926 (-1.88) |
| (c) importance sampling (50 % in r<0.3) | 9.82, 1.87, 1.39, 2.50, 5.35 | 12.5 | 12.301 (+1.20) | 12.132 (-0.19) | 12.00–12.36 (-1.29 to +1.72) | 0.83 | 0.455 | – | 17 | 12.075 (-0.66) |
| (d) mu = 1000 | 11.07, 2.69, 2.21, 2.99, 5.79 | 12.5 | 12.379 (+1.84) | 12.093 (-0.50) | 12.00–12.34 (-1.29 to +1.56) | 0.82 | 0.552 | – | 17 | 12.050 (-0.86) |
| (d) 5000 epochs | 2.19, 0.68, 1.97 | 12.5 | 12.540 (+3.17) | 12.186 (+0.26) | 12.01–12.39 (-1.22 to +1.93) | 0.74 | 0.258 | – | 32 | 12.117 (-0.31) |
| (a) mixed, lam=100, seed 0 | 435.23, 2.28, 1.65, 2.98, 6.05 | 12.5 | 12.323 (+1.38) | 12.198 (+0.36) | 11.98–12.40 (-1.47 to +2.04) | 0.00 | 0.008 | [10.5] | 9 | 12.112 (-0.35) |
| (a) mixed, lam=100, seed 1 | 21.14, 2.63, 1.83, 3.06, 6.07 | 12.5 | 12.394 (+1.97) | 12.191 (+0.30) | 11.96–12.43 (-1.59 to +2.23) | 0.02 | 0.007 | – | 8 | 12.099 (-0.46) |
| (a) mixed, lam=100, seed 2 | 18.40, 3.55, 1.93, 3.12, 6.24 | 12.5 | 12.576 (+3.47) | 12.183 (+0.23) | 11.88–12.38 (-2.24 to +1.88) | 0.01 | 0.007 | – | 8 | 12.046 (-0.89) |
| (a) mixed, lam=10, seed 0 | 1.95, 0.56, 0.36, 1.30, 3.34 | 12.5 | 12.171 (+0.14) | 12.304 (+1.23) | 11.86–12.79 (-2.45 to +5.20) | 0.00 | 0.004 | – | 9 | 12.145 (-0.08) |
| (a) mixed, lam=10, seed 1 | 1.95, 0.56, 0.38, 1.30, 3.35 | 12.5 | 12.161 (+0.06) | 12.251 (+0.80) | 11.86–12.80 (-2.41 to +5.35) | 0.00 | 0.004 | – | 9 | 12.093 (-0.51) |
| (a) mixed, lam=10, seed 2 | 1.92, 0.54, 0.33, 1.28, 3.33 | 12.5 | 12.181 (+0.22) | 12.311 (+1.28) | 11.85–12.77 (-2.48 to +5.07) | 0.00 | 0.001 | – | 8 | 12.151 (-0.03) |
| (a) mixed, lam=1000 | 3e+04, 4e+04, 3e+03, 155.31, 122.36 | 14.5 | 14.012 (+15.28) | 13.035 (+7.24) | 10.18–10.18 (-16.21 to -16.21) | 0.00 | 7.935 | [10.5, 11.5, 12.5] | 9 | nan (+nan) |
| (a) mixed, lam=100, P0 | 8e+04, 5.39, 1.62, 3.86, 6.94 | 12.5 | 12.627 (+3.89) | 12.107 (-0.39) | 12.11–12.40 (-0.39 to +2.02) | 0.03 | 0.006 | [10.5] | 6 | 12.123 (-0.26) |
| (b) down-weighting omega=|Du|^2/(|Du|^2+0.09) | 8.34, 2.07, 1.20, 2.05, 5.09 | 12.5 | 12.506 (+2.89) | 12.187 (+0.27) | 12.09–12.39 (-0.57 to +1.96) | 0.77 | 0.315 | – | 19 | 12.152 (-0.02) |
| (b') divide by (|Du|^2+0.09)^{(p-2)/2} | 20.55, 20.42, 16.88, 12.83, 11.58 | 14.5 | 14.445 (+18.84) | 10.296 (-15.29) | 9.47–12.68 (-22.07 to +4.34) | 0.92 | 10.734 | – | 18 | nan (+nan) |
| (e) singular feature |x|^{p/(p-1)}, seed 0 | 5.20, 0.64, 0.22, 1.67, 4.87 | 12.5 | 12.224 (+0.57) | 12.241 (+0.71) | 12.05–12.44 (-0.90 to +2.33) | 0.31 | 0.027 | – | 16 | 12.178 (+0.19) |
| (e) singular feature, seed 1 | 2.55, 0.44, 0.15, 1.64, 4.88 | 12.5 | 12.162 (+0.06) | 12.209 (+0.45) | 12.07–12.37 (-0.68 to +1.77) | 0.28 | 0.012 | – | 15 | 12.163 (+0.07) |
| (e) singular feature, seed 2 | 19.81, 0.69, 0.21, 1.65, 4.86 | 12.5 | 12.251 (+0.79) | 12.188 (+0.27) | 12.01–12.39 (-1.20 to +1.95) | 0.32 | 0.036 | – | 16 | 12.120 (-0.29) |
| (e) singular feature, trainable centre | 5.19, 0.62, 0.22, 1.67, 4.88 | 12.5 | 12.215 (+0.50) | 12.241 (+0.71) | 12.04–12.44 (-0.92 to +2.35) | 0.30 | 0.026 | – | 16 | 12.177 (+0.18) |

### Table 2: three-seed summary (node 12.5)

| variant | loss(12.5) per seed | E_PINN(12.5) per seed | err % | 3-pt vertex per seed | err % | self-consistent E* per seed | err % |
|---|---|---|---|---|---|---|---|
| baseline P1 | 2.049, 1.669, 2.174 | 12.108, 12.147, 12.112 | -0.38, -0.06, -0.35 | 12.331, 12.316, 12.659 | +1.45, +1.33, +4.15 | 12.064, 12.075, 12.024 | -0.75, -0.65, -1.07 |
| baseline P0 | 1.500, 1.230, 1.612 | 12.191, 12.180, 12.145 | +0.30, +0.21, -0.08 | 12.352, 12.282, 12.405 | +1.63, +1.05, +2.06 | 12.145, 12.117, 12.072 | -0.08, -0.31, -0.68 |
| (a) mixed lam=100 | 1.646, 1.832, 1.926 | 12.198, 12.191, 12.183 | +0.36, +0.30, +0.23 | 12.323, 12.394, 12.576 | +1.38, +1.97, +3.47 | 12.112, 12.099, 12.046 | -0.35, -0.46, -0.89 |
| (a) mixed lam=10 | 0.361, 0.380, 0.335 | 12.304, 12.251, 12.311 | +1.23, +0.80, +1.28 | 12.171, 12.161, 12.181 | +0.14, +0.06, +0.22 | 12.145, 12.093, 12.151 | -0.08, -0.51, -0.03 |
| (e) singular feature | 0.221, 0.150, 0.210 | 12.241, 12.209, 12.188 | +0.71, +0.45, +0.27 | 12.224, 12.162, 12.251 | +0.57, +0.06, +0.79 | 12.178, 12.163, 12.120 | +0.19, +0.07, -0.29 |

### Table 3: fine scans (h = 0.25, E = 11 … 14, warm start, seed 0)

| variant | loss at the 13 nodes | argmin (err %) | nodes within 25 % of the minimum | E_PINN at argmin (err %) | MC std @argmin | self-consistent E* (err %) |
|---|---|---|---|---|---|---|
| baseline P0 | 4.50, 2.90, 1.57, 1.46, 1.08, 1.33, 1.30, 0.92, 1.36, 2.50, 2.50, 2.94, 4.26 | 12.75 (+4.90) | 12.0–12.75 | 12.233 (+0.65) | 0.340 | 12.103 (-0.42) |
| baseline P1 | 8.70, 2.76, 2.12, 1.22, 0.67, 0.59, 0.71, 0.67, 1.06, 1.56, 2.03, 2.60, 3.52 | 12.25 (+0.78) | 12.0–12.75 | 12.160 (+0.04) | 0.369 | 12.139 (-0.13) |
| (a) mixed lam=100 P1 | 88.62, 2.57, 1.76, 1.35, 1.21, 1.10, 1.21, 1.39, 1.60, 2.06, 2.52, 3.05, 3.74 | 12.25 (+0.78) | 11.75–12.5 | 12.159 (+0.04) | 0.003 | 12.120 (-0.28) |
| (e) singular feature P1 | 3.96, 0.96, 0.48, 0.21, 0.08, 0.05, 0.16, 0.42, 0.69, 1.09, 1.61, 2.23, 3.01 | 12.25 (+0.78) | 12.25–12.25 | 12.217 (+0.51) | 0.005 | 12.191 (+0.30) |

### Table 4: second eigenvalue window (reference 69.160, nodes from scratch)

| run | nodes | loss | mass | E_PINN |
|---|---|---|---|---|
| base_E2 | [65.0, 67.0, 69.0, 71.0, 73.0] | 2.33e+03, 2.14e+03, 881, 1.76e+03, 190 | 0.007, 0.009, 0.060, 0.013, 0.476 | 31.13, 38.31, 62.52, 50.08, 69.31 |
| mixed_E2 | [65.0, 67.0, 69.0, 71.0, 73.0] | 2.64e+03, 2.84e+03, 3.02e+03, 3.2e+03, 3.46e+03 | 0.005, 0.004, 0.004, 0.003, 0.003 | 14.89, 15.12, 15.25, 15.47, 15.71 |
| base_E2_premass | [65.0, 67.0, 69.0, 71.0, 73.0] | 2.42e+03, 1.51e+03, 399, 1.61e+03, 178 | 0.006, 0.017, 0.227, 0.016, 0.501 | 27.46, 48.38, 67.69, 53.73, 69.39 |
| mixed_E2_premass | [65.0, 67.0, 69.0, 71.0, 73.0] | 2.78e+03, 2.83e+03, 3.03e+03, 3.23e+03, 3.42e+03 | 0.004, 0.004, 0.004, 0.003, 0.003 | 13.65, 15.22, 15.19, 15.44, 15.43 |

### Table 5: sources of fluctuation of the loss at E = 12.5 (baseline strong form)

| source | measure | value |
|---|---|---|
| training-batch residual, last 30 recorded epochs, P1 (lr decay) | mean ± std (batch of 5000, moving network) | 1.65 ± 0.52 |
| same network frozen, P1 (lr decay) | std over 8 fresh batches of 5000 / value on 40000 | 0.532 / 2.049 (range 1.51–2.97) |
| training-batch residual, last 30 recorded epochs, constant lr | mean ± std (batch of 5000, moving network) | 1.43 ± 0.52 |
| same network frozen, constant lr | std over 8 fresh batches of 5000 / value on 40000 | 0.522 / 1.749 (range 1.25–2.58) |
| training-batch residual, last 30 recorded epochs, P0 (const lr + early stop) | mean ± std (batch of 5000, moving network) | 1.65 ± 0.56 |
| same network frozen, P0 (const lr + early stop) | std over 8 fresh batches of 5000 / value on 40000 | 0.588 / 1.500 (range 0.77–2.37) |
| training-batch residual, last 30 recorded epochs, mixed P1 | mean ± std (batch of 5000, moving network) | 0.35 ± 0.01 |
| same network frozen, mixed P1 | std over 8 fresh batches of 5000 / value on 40000 | 0.008 / 0.358 (range 0.35–0.37) |
| training-batch residual, last 30 recorded epochs, singular feature P1 | mean ± std (batch of 5000, moving network) | 0.23 ± 0.06 |
| same network frozen, singular feature P1 | std over 8 fresh batches of 5000 / value on 40000 | 0.027 / 0.221 (range 0.19–0.27) |
| seed-to-seed, P1 | loss(12.5) for 3 seeds | 2.05, 1.67, 2.17 |
| seed-to-seed, P0 | loss(12.5) for 3 seeds | 1.50, 1.23, 1.61 |
| amount of training, 1500 epochs at the node (h=1 scan) | loss(12.5), core frac | 2.05, 0.81 |
| amount of training, 5000 epochs at the node | loss(12.5), core frac | 0.68, 0.74 |
| amount of training, h=0.25 warm-started scan (6 previous nodes x 1500 epochs nearby) | loss(12.5), core frac | 0.71, 0.79 |
