# p-Laplacian experiments (agent E) — answer to Reviewer #1, point 5

Everything here uses `pinn_eigen.py` unchanged, with the **corrected** loss

    Λ̄_p(u,E) = ∫(Δ_p u + E|u|^{p−2}u)² dx + μ(∫|u|^p dx − 1)²      (integrals = |Ω|·MC mean, `vol=|Ω|`)

except the runs explicitly marked *legacy* (`legacy_plap=True`: residual `Δ_p u + E·u`, normalisation `mean(u²)=1`,
μ = 4000), which is the loss that produced the published Tables 5–6 and Fig. 16.

## Files

| file | content |
|---|---|
| `refs.py` | radial shooting, unit ball of R^d: `E_k = z_k^p`, z_k = k-th zero of the E=1 solution → `results/plap_refs.json` |
| `fem_plap.py` | independent P1 FEM for *first* eigenvalues (Rayleigh-quotient minimisation, L-BFGS, float64, 4 nested meshes + Richardson) → `results/plap_fem_refs.json` |
| `core.py`, `jobs.py`, `run.py` | resumable PINN scans (one JSON per job: `results/plap_<job>.json`, one record per E, written after every E; last network in `ckpt/`) |
| `analyze.py`, `make_tex.py` | tables (`tables_auto.md`, `results/plap_summary.json`), figures, `tables.tex` |
| `figures/` | `plap_disk_curves.png`, `plap_square_curves.png`, `plap_ball_curves.png`, `plap_convergence_curves.png` |
| `logs/` | raw logs (two container restarts happened; all jobs were resumed from the JSON records) |

Re-run: `python -m experiments.plap.refs; python -m experiments.plap.fem_plap --nmax 160;
python -m experiments.plap.run --worker 0 (and 1); python -m experiments.plap.analyze; python -m experiments.plap.make_tex`.

## Settings

* Network 3×128 tanh times B(x); N = 5000 fresh MC points per epoch; Adam; μ = 200 for every corrected-loss run; seed 0
  (seeds 1, 2 in the convergence study); 1 torch thread per process. Per-E seeds are `seed·100003 + i` (deterministic on resume).
* **P1 (new protocol)**: lr 1e-3 → 1e-5 cosine decay over 1500 epochs for every E, no early stopping, warm start along the grid.
* **P0 (protocol of the paper)**: lr 1e-3 constant, ≤ 2000 epochs, early stopping (patience 300, tol 1e-5), warm start.
* Coarse grid h = 1 (2-D) or h = 2 (3-D ball); then, at every detected local minimum, a local 3-point scan with h = 0.5 (1.0 in 3-D),
  warm-started from the coarse network when it was stored (otherwise from scratch, 3000 epochs for the first point).
* μ: the theory rule (Prop. `prop.penalized`: no collapse to u ≡ 0 as long as Λ(E) ≤ 2μ) needs μ ≥ ½·max Λ over the window. The largest
  value on our windows is Λ ≈ 54 in 2-D (disk, E = 50) and ≈ 90 in 3-D (ball, E = 46), so μ = 200 satisfies it everywhere (factor ≥ 4); μ ∈ {50, 1000} change the
  refined eigenvalue by ≤ 0.05 % (convergence table).
* Every number called Λ̄ is evaluated on a **fresh** sample of 40 000 points (not the last training batch) and divided by
  (∫|u|^p)^{2(p−1)/p} (scale invariant; the mass is 1 ± 0.02 anyway). r = Λ̄^{1/2}.
* Three eigenvalue estimators are recorded for each trained network:
  1. `E_curve`: vertex of the parabola through the lowest grid point of the loss curve and its two neighbours;
  2. `E_ls` (reported as **E_PINN**): minimiser in E of the residual of the *frozen* trained network, in closed form
     `E_ls = −⟨Δ_p u, w⟩/⟨w, w⟩`, w = |u|^{p−2}u. This is exactly the "refinement" step of the paper (there done with a noisy
     `minimize_scalar`), evaluated on 40 000 fresh points;
  3. the Rayleigh quotient ∫|∇u|^p/∫|u|^p (kept in the JSON; with 40 000 points its MC error is ≈ ±0.5–1 %, larger than the error of
     `E_ls`, so it is not used).
* Cost: 2-D ≈ 0.14–0.17 s/epoch (1 thread, machine shared) → ≈ 3.5–4 min per E; 3-D ≈ 0.18–0.25 s/epoch. Total training wall time of
  all jobs: see the last line of the tables (≈ 15 core-hours); FEM references ≈ 25 min; shooting < 10 s.

## 1. References

* **Shooting** (`refs.py`): E_1 of the unit ball for p ∈ {1.5, 2, 2.2, 2.5, 3, 4}, d ∈ {2, 3}; independent of (r0, tol) to ≤ 4·10⁻¹⁰;
  p = 2 reproduces j_{0,1}² = 5.783186 and π²; (p,d) = (2.2,2): 6.53174 (Horák: 6.5320); (2.5,3): 14.11123.
  Higher zeros give the radial (nodal-circle) eigenvalues, e.g. 41.7886 for (2.2, 2).
* **FEM** (`fem_plap.py`): first eigenvalue of a domain = global minimum of the Rayleigh quotient; for positive u the quotient has no other
  critical point, so descent from a positive start is safe. Observed order 2.0 on all domains; validation: disk p=2 5.78319 (exact 5.78319),
  disk p=2.2 6.53174 (= shooting), p = 1.5/3/4 4.01780/9.83150/14.68163 (shooting 4.01780/9.83150/14.68164), half disk p=2 14.68197
  (j_{1,1}² = 14.68197), quarter disk p=2 26.37462 (j_{2,1}² = 26.37462), axisymmetric ball p=2.5 14.11123 (= shooting),
  half ball p=2 20.19073 (exact 20.19073), unit square p=2.2 25.24871 = 2^{2.2}·5.49505 (scaling, to 6 digits).
* **Odd reflection.** If v is the first eigenfunction of a subdomain whose odd reflections tile Ω, the reflected function is an eigenfunction
  of Ω with the same E. This gives eigenvalues of the disk (sectors of angle π/m: m nodal diameters) and of the square (rectangle 1×2,
  diagonal triangle, 2×2 cells, the four triangles cut by both diagonals, three strips).
  **These are eigenvalues with a known eigenfunction symmetry; nothing is claimed about their position in the
  Lusternik–Schnirelmann (variational) ordering**, except for E_1 and for the second eigenvalue (Horák's values 18.395 / 15.144, which our
  FEM reproduces as λ_1(half disk) = 18.3929 and λ_1(diagonal triangle) = 15.1429; the midline rectangle gives 15.2808, i.e. at p = 2.2 the
  double eigenvalue 5π²/4 of the Laplacian splits and the diagonal mode is the lower one). The scan detects whichever eigenvalues lie in
  the window; the column "minimum k" below is the order of detection, not an index.

## 2. Main findings

1. **The published values were not p-Laplacian eigenvalues.** With the legacy loss the frozen-network estimate is E_ls = 6.93–6.95 on the
   disk for every training E (the equation −Δ_p u = E u with mean(u²) = 1 has the radial solution E = 6.935), and the P0 loss curve is flat
   and noisy between 6.25 and 7.25 (values 0.46–1.4), so a local minimum anywhere in 6.5–7.0 is possible: the published 6.679 is of this
   type. In 3-D (p = 2.5) the legacy loss gives 17.62 (E_ls) / 17.82 (curve vertex): the published 17.4513, 24–26 % above the true
   E_1 = 14.1112. The legacy loss therefore explains the old numbers; they must be replaced.
2. **With the corrected loss the method works, including for higher eigenvalues.** Disk p = 2.2, window [3,58]: five minima,
   E_PINN = 6.5337, 18.4069, 34.8043, 41.96, 56.1166 against 6.5317, 18.3929, 34.8082, 41.7886, 56.1099 (errors +0.03, +0.08, −0.01,
   +0.41, +0.01 %). Square [0,2]², window [3,37]: 5.4902, 15.1440, 25.2285, 32.4615 against 5.4950, 15.1429, 25.2487, 32.4595
   (−0.09, +0.01, −0.08, +0.01 %). 3-D ball p = 2.5, window [8,46]: 14.1004 and 35.6283 against 14.1112 (shooting) and 35.6315
   (axisymmetric FEM, half ball) (−0.08, −0.01 %). The 3rd and higher values are, to our knowledge, new data; each is matched by an
   independent reference.
   Caveats: (i) nearby eigenvalues are not separated — the square has 15.1429 (diagonal) and 15.2808 (midline) inside one minimum
   (the network trained at E ≤ 16 gives E_ls = 15.144, at E = 17–19 it drifts to 15.24–15.31), and 32.46 / 33.17 (three strips) likewise;
   (ii) the radial nodal-circle mode of the disk (41.79) has by far the largest floor (Λ̄ ≈ 4.7, r ≈ 2.2) and the largest error
   (+0.41 % for E_ls, +1.6…2.2 % for the curve vertex): its gradient vanishes on a whole circle, not only at isolated points.
3. **Dominant error sources (disk, p = 2.2, E_1; table "Convergence study").**
   * *Optimisation noise of the last iterate (dominant in the published protocol).* With constant lr and early stopping (P0) the loss at
     a fixed E fluctuates by ±0.1–0.2 between seeds/epochs (Λ̄(6.5) = 0.35, 0.20, 0.27 for three seeds; 0.13, 0.13, 0.125 with lr decay),
     while the curve only rises by ≈ 0.27 at distance 0.5 from the eigenvalue (curvature ≈ 1.07 = ∫|u_1|^{2(p−1)}). The minimum of a noisy
     curve of this kind is located with an error of several tenths: the 3-point vertex has errors −0.17, +0.47, **+2.26 %** over three
     seeds with P0, against +0.18, +0.20, +0.19 % with lr decay. This is the size of the errors in the old tables (2–6 %).
   * *Residual floor at the critical points (dominant after lr decay).* The loss does not go to 0 at the eigenvalue: Λ̄(E_1) ≈ 0.12–0.13,
     and 50–60 % of it comes from the 2 % of the sample points with the smallest |∇u| (column "core frac"; 0.5 for E_1 of the 3-D ball,
     0.83 for p = 3). This is the defect predicted by Lemma `lem.critical`: a smooth network has Δ_p u = 0 at its critical points while
     E|u|^{p−2}u ≠ 0 there. The floor decreases only slowly with capacity and training: 0.150 (2×32), 0.137 (2×64), 0.131 (3×128),
     0.096 (4×256); 0.153 (500 epochs), 0.131 (1500), 0.049 (5000). It is not symmetric in E, so it biases the minimum of the curve
     upwards by a stable +0.2…0.8 %, and this bias is *not* removed by a larger network (4×256: +0.41 %), more points (N = 20 000:
     +0.10 %), or longer training (5000 epochs: +0.48 %).
   * *E-grid step.* Refining the grid makes the curve estimate **worse**, not better: h = 0.5 → +0.18 %, h = 0.25 → +0.78 %,
     h = 0.125 → +3.7 %, because the differences of Λ̄ between neighbouring grid points (≈ c h²) fall below the floor fluctuations
     (≈ 0.02 with lr decay). The step should satisfy c h² ≫ δΛ, i.e. h ≳ 0.5 here; a coarse grid plus the frozen-network refinement is both
     cheaper and more accurate than a fine grid.
   * *Monte Carlo sampling* is a minor contribution at fixed protocol: N = 2000 / 5000 / 20 000 give floors 0.127 / 0.131 / 0.123 and
     E_ls errors +0.13 / ±0.05 (3 seeds) / 0.00 %.
   * *Network capacity (approximation)* is a minor contribution for p = 2.2: E_ls errors +0.05 % (2×32, 1 min), +0.04 % (2×64),
     ≤ 0.05 % (3×128), +0.03 % (4×256, 36 min).
   * *Penalty*: μ = 50 / 200 / 1000 → E_ls errors 0.00 / 0.00 / +0.05 %.
   * The refined estimate E_ls is insensitive to all of the above (|error| ≤ 0.05 % for every lr-decay configuration with ≥ 1500 epochs
     and N ≥ 5000; 0.12–0.37 % with P0) and even to the training value of E (6.532–6.538 for E between 5.5 and 7.5): it is a quotient computed from the
     trained function alone, whereas the position of the minimum of the loss curve inherits the E-dependence of the floor.
4. **p far from 2 (disk, E_1, baseline settings).** p = 3: E_ls = 9.801 (−0.31 %), curve vertex +0.55 %, floor 0.60, core fraction 0.83.
   p = 4: floor ≈ 6–8, curve essentially flat between E = 14 and 17 (vertex +8 %), E_ls between 14.34 and 15.16 depending on the training E
   (+1.5 % at the grid minimum): the critical-point defect grows with p (the eigenfunction is only C^{1,1/(p−1)} at its maximum) and
   becomes the limiting factor; accuracy degrades from 0.05 % (p = 2.2) to 0.3 % (p = 3) and 1–2 % (p = 4). A 4×256 network started from
   scratch at p = 4 collapsed to u ≡ 0 (mass 0; for p > 2 the trivial function is a degenerate critical point of the loss) — one run,
   interrupted by a container restart and not repeated. **p = 1.5 fails**: Λ̄ = 24–58, no minimum near E_1 = 4.018 (E_ls = 5.3–7.8).
   For p < 2 the strong form Δ_p u of a smooth network is singular (|x−x_0|^{p−2}) at its critical points and the MC estimator of the
   loss has infinite variance for p ≤ 3/2 in d = 2 (Remark `rem.plap.errors`(2)); the method in its present (strong-form, unregularised)
   version should be claimed for p ≥ 2 only.
5. **Recommended text for the paper.** (a) Replace Tables 5–6 and Fig. 16 by the tables in `tables.tex` and the figures here;
   (b) state the training protocol with lr decay and the closed-form refinement; (c) say that the remaining error is controlled by the
   residual floor created at the critical points of the network (Lemma `lem.critical`), that it grows with p and with the size of the
   critical set, and that for p-Laplacian problems the loss-curve minimum should be used for *detection* and the frozen-network
   quotient for the *value*; (d) grid refinement below h ≈ (δΛ/c)^{1/2} is counter-productive.

## 3. Limitations / things not done

* One seed for all runs except the baseline and P0 convergence rows (3 seeds each).
* The local 3-point scans around the disk E_1/E_2 and square E_1 minima and both 3-D minima were started from scratch (coarse networks were
  not stored for the first coarse jobs); for the second eigenvalue of the 3-D ball this gave a poor local curve (vertex 36.6, +2.8 %,
  floor 3.4) although E_ls = 35.628 is accurate — the coarse-curve vertex (35.64) is the better curve estimate there.
* `disk_p4_w256d4` is incomplete (1 of 3 points, collapsed) and is excluded from the tables.
* The reference for the second eigenvalue of the 3-D ball (35.6315) is λ_1 of the half ball from an axisymmetric P1 FEM; it is an
  eigenvalue of the ball with one nodal plane and an upper bound for λ_2; that it *is* λ_2 is not proved.
* No LaTeX compilation of `tables.tex` was done (booktabs syntax only; it cites `\cite{HJ11}`).
* Timings in the tables were taken on a shared machine (2–4 concurrent processes) and are indicative only.

## 4. Tables (generated by `analyze.py`; errors in %)

### References

| p | d | E1(B_1) shooting | 2nd radial |
|---|---|---|---|
| 1.5 | 2 | 4.017795 | 13.0737 |
| 2.0 | 2 | 5.783186 | 30.4713 |
| 2.2 | 2 | 6.531740 | 41.7886 |
| 2.5 | 2 | 7.710246 | 66.1125 |
| 3.0 | 2 | 9.831498 | 137.9339 |
| 4.0 | 2 | 14.681640 | 559.0275 |
| 1.5 | 3 | 6.371403 | 16.6013 |
| 2.0 | 3 | 9.869604 | 39.4784 |
| 2.2 | 3 | 11.469906 | 54.6063 |
| 2.5 | 3 | 14.111228 | 87.4923 |
| 3.0 | 3 | 19.201037 | 186.2603 |
| 4.0 | 3 | 32.214611 | 782.8237 |

| FEM (P1, Richardson) | p | finest mesh | extrapolated | order |
|---|---|---|---|---|
| disk | 2.0 | 5.78328 | 5.78319 | 2.00 |
| disk | 2.2 | 6.53187 | 6.53174 | 2.00 |
| halfdisk | 2.0 | 14.68272 | 14.68197 | 2.00 |
| halfdisk | 2.2 | 18.39393 | 18.39286 | 2.00 |
| square2 | 2.2 | 5.49519 | 5.49505 | 1.98 |
| rect1x2 | 2.2 | 15.28162 | 15.28076 | 1.98 |
| tri_diag | 2.2 | 15.14373 | 15.14289 | 2.00 |
| ball3_axi | 2.5 | 14.11159 | 14.11123 | 2.00 |
| halfball3_axi | 2.0 | 20.19206 | 20.19073 | 2.00 |
| halfball3_axi | 2.5 | 35.63469 | 35.63150 | 2.00 |
| disk | 1.5 | 4.01784 | 4.01780 | 2.00 |
| disk | 3.0 | 9.83182 | 9.83150 | 1.99 |
| disk | 4.0 | 14.68246 | 14.68163 | 1.99 |
| sector_pi/2 | 2.0 | 26.37724 | 26.37462 | 2.00 |
| sector_pi/2 | 2.2 | 34.81212 | 34.80823 | 2.00 |
| sector_pi/3 | 2.2 | 56.11997 | 56.10990 | 2.00 |
| sector_pi/4 | 2.2 | 82.30762 | 82.28724 | 2.03 |
| strip3 | 2.2 | 33.17340 | 33.17197 | 2.00 |
| unit_square | 2.2 | 25.25120 | 25.24871 | 2.02 |

### Corrected loss (protocol P1): detected eigenvalues

| case | reference (type) | coarse 3-pt vertex | err % | local 3-pt vertex | err % | E_ls | err % | Lam_min (fresh) | r=sqrt | core frac |
|---|---|---|---|---|---|---|---|---|---|---|
| disk p=2.2, window [3,58], minimum 1 | 6.5317 (radial shooting (1st eigenvalue)) | 6.550 | +0.28 | 6.554 | +0.34 | 6.5337 | +0.03 | 0.115 | 0.340 | 0.56 |
| disk p=2.2, window [3,58], minimum 2 | 18.3929 (FEM half disk (1 nodal diameter); Horak 18.395) | 18.433 | +0.22 | 18.414 | +0.11 | 18.4069 | +0.08 | 0.832 | 0.912 | 0.40 |
| disk p=2.2, window [3,58], minimum 3 | 34.8082 (FEM sector pi/2 (2 nodal diameters)) | 34.848 | +0.11 | 34.794 | -0.04 | 34.8043 | -0.01 | 0.783 | 0.885 | 0.56 |
| disk p=2.2, window [3,58], minimum 4 | 41.7886 (radial shooting, 1 nodal circle) | 42.697 | +2.17 | 42.454 | +1.59 | 41.9615 | +0.41 | 4.734 | 2.176 | 0.34 |
| disk p=2.2, window [3,58], minimum 5 | 56.1099 (FEM sector pi/3 (3 nodal diameters)) | 56.170 | +0.11 | 55.658 | -0.81 | 56.1166 | +0.01 | 1.656 | 1.287 | 0.55 |
| square p=2.2, window [3,37], minimum 1 | 5.4950 (FEM square (1st eigenvalue); Horak 5.4952) | 5.511 | +0.28 | 5.522 | +0.49 | 5.4902 | -0.09 | 0.217 | 0.466 | 0.31 |
| square p=2.2, window [3,37], minimum 2 | 15.1429 (FEM half square, diagonal nodal line; Horak 15.144) | 15.146 | +0.02 | 15.160 | +0.11 | 15.1440 | +0.01 | 0.197 | 0.444 | 0.11 |
| square p=2.2, window [3,37], minimum 3 | 25.2487 (FEM [0,1]^2 = 2^p E1 (2x2 cells)) | 25.232 | -0.07 | 25.319 | +0.28 | 25.2285 | -0.08 | 1.026 | 1.013 | 0.58 |
| square p=2.2, window [3,37], minimum 4 | 32.4595 (scaled FEM triangle (both diagonals nodal), 2^{p/2} x 15.1429) | 32.509 | +0.15 | 32.544 | +0.26 | 32.4615 | +0.01 | 0.935 | 0.967 | 0.03 |
| ball3 p=2.5, minimum 1 | 14.1112 (radial shooting (1st eigenvalue)) | 14.133 | +0.15 | 14.215 | +0.73 | 14.1004 | -0.08 | 0.399 | 0.632 | 0.53 |
| ball3 p=2.5, minimum 2 | 35.6315 (FEM axisymmetric half ball (nodal plane)) | 35.640 | +0.02 | 36.627 | +2.79 | 35.6283 | -0.01 | 3.433 | 1.853 | 0.11 |

### Paper protocol (P0) and legacy loss

| run | grid min | 3-pt vertex | E_ls | Lam at min | published | true | err % |
|---|---|---|---|---|---|---|---|
| disk p=2.2 correct loss, P0 (paper protocol) | 6.00 | 6.4949 | 6.5309 | 0.401 | -- | 6.5317 | -0.56 |
| disk p=2.2 correct loss, P0 (paper protocol) | 18.00 | 18.4019 | 18.3835 | 0.687 | -- | 18.3929 | +0.05 |
| square p=2.2 correct loss, P0 | 6.00 | 5.5035 | 5.5021 | 0.345 | -- | 5.4950 | +0.15 |
| square p=2.2 correct loss, P0 | 15.00 | 15.1478 | 15.1423 | 0.330 | -- | 15.1429 | +0.03 |
| disk p=2.2 LEGACY loss, P0 | 7.00 | 6.8753 | 6.9241 | 0.460 | 6.679 | 6.5317 | +5.26 |
| disk p=2.2 LEGACY loss, P1 | 7.00 | 6.9560 | 6.9342 | 0.365 | 6.679 | 6.5317 | +6.50 |
| ball3 p=2.5 LEGACY loss, P1 | 18.00 | 17.8199 | 17.6165 | 1.937 | 17.4513 | 14.1112 | +26.28 |

### Convergence study, disk p=2.2, E1 = 6.53174 (grid 6.0, 6.5, 7.0 unless stated)

| setting | Lam(6.0) | Lam(6.5) | Lam(7.0) | core frac | 3-pt vertex | err % | E_ls | err % | RQ | err % | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline P1: 3x128, N=5000, 1500 ep, mu=200, h=0.5 (seed 0) | 0.394 | 0.131 | 0.316 | 0.59 | 6.5433 | +0.18 | 6.5320 | +0.00 | 6.5864 | +0.84 | 18 |
| baseline, seed 1 | 0.416 | 0.130 | 0.329 | 0.54 | 6.5449 | +0.20 | 6.5329 | +0.02 | 6.4834 | -0.74 | 10 |
| baseline, seed 2 | 0.401 | 0.125 | 0.318 | 0.56 | 6.5443 | +0.19 | 6.5285 | -0.05 | 6.5938 | +0.95 | 10 |
| P0 (constant lr, early stop), seed 0 | 0.439 | 0.348 | 0.426 | 0.20 | 6.5204 | -0.17 | 6.5393 | +0.12 | 6.5570 | +0.39 | 5 |
| P0, seed 1 | 0.427 | 0.204 | 0.338 | 0.29 | 6.5623 | +0.47 | 6.5453 | +0.21 | 6.5349 | +0.05 | 8 |
| P0, seed 2 | 0.419 | 0.272 | 0.296 | 0.19 | 6.6795 | +2.26 | 6.5557 | +0.37 | 6.4468 | -1.30 | 6 |
| network 2x32 | 0.534 | 0.150 | 0.353 | 0.40 | 6.5767 | +0.69 | 6.5350 | +0.05 | 6.5249 | -0.10 | 1 |
| network 2x64 | 0.467 | 0.137 | 0.348 | 0.46 | 6.5550 | +0.36 | 6.5343 | +0.04 | 6.5250 | -0.10 | 3 |
| network 4x256 | 0.436 | 0.096 | 0.307 | 0.48 | 6.5586 | +0.41 | 6.5338 | +0.03 | 6.5243 | -0.11 | 36 |
| N=2000 | 0.417 | 0.127 | 0.332 | 0.50 | 6.5431 | +0.17 | 6.5403 | +0.13 | 6.5938 | +0.95 | 4 |
| N=20000 | 0.397 | 0.123 | 0.325 | 0.61 | 6.5380 | +0.10 | 6.5319 | +0.00 | 6.4876 | -0.68 | 40 |
| 500 epochs | 0.515 | 0.153 | 0.338 | 0.35 | 6.5812 | +0.76 | 6.5415 | +0.15 | 6.5476 | +0.24 | 3 |
| 5000 epochs | 0.376 | 0.049 | 0.244 | 0.47 | 6.5633 | +0.48 | 6.5315 | -0.00 | 6.5865 | +0.84 | 32 |
| mu=50 | 0.390 | 0.073 | 0.250 | 0.55 | 6.5712 | +0.60 | 6.5317 | -0.00 | 6.5240 | -0.12 | 10 |
| mu=1000 | 0.516 | 0.166 | 0.385 | 0.31 | 6.5575 | +0.39 | 6.5352 | +0.05 | 6.5249 | -0.10 | 10 |
| E-step h=0.25 | -- | 0.120 | -- | 0.56 | 6.5828 | +0.78 | 6.5338 | +0.03 | 6.5248 | -0.11 | 10 |
| E-step h=0.125 | -- | 0.116 | -- | 0.55 | 6.7727 | +3.69 | 6.5337 | +0.03 | 6.5247 | -0.11 | 10 |

### Other p (unit disk, E1, baseline P1)

| p | run | h | ref | 3-pt vertex | err % | E_ls | err % | RQ | err % | Lam_min | core frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | disk_p1.5 | 0.5 | 4.0178 | 4.7024 | +17.04 | 5.2887 | +31.63 | 5.2374 | +30.36 | 24.197 | 0.36 |
| 2.2 | conv_base_s0 | 0.5 | 6.5317 | 6.5433 | +0.18 | 6.5320 | +0.00 | 6.5864 | +0.84 | 0.131 | 0.59 |
| 3.0 | disk_p3 | 0.75 | 9.8315 | 9.8851 | +0.55 | 9.8012 | -0.31 | 9.9146 | +0.85 | 0.603 | 0.83 |
| 4.0 | disk_p4 | 1.0 | 14.6816 | 15.8859 | +8.20 | 14.8952 | +1.45 | 14.7695 | +0.60 | 6.262 | 0.66 |

Total training wall time (sum over jobs, 1 thread each): 14.5 h
