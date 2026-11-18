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

---

# Round 2 (referee check H, items 9 and 13; authors' request on the regularized operator)

Code: jobs `r2_*` and `reg_*` in `jobs.py` (queues 2–4), `refs_reg.py`, `analyze_r2.py` → `tables_r2.md`, `tables_reg.md`,
`results/plap_summary_r2.json`, `results/plap_refs_reg.json`; LaTeX appended to `tables.tex` (Tables `tab.plap.seeds`,
`tab.plap.conv2`, `tab.plap.p.seeds`, `tab.plap.reg`). One process, 1 thread; ≈ 5.5 h of training in total. Same baseline settings as
round 1 (3×128, N = 5000, μ = 200, P1). **Seeds: every round-1 number is seed 0** unless a seed is named; round 2 adds seeds 1 and 2.

## R2.1 Seeds for the main p = 2.2 results (item 9)

Local 3-node scans (h = 0.5), three seeds each (table (a) below).
* Disk E_1: E_PINN = 6.5320, 6.5329, 6.5285 → mean 6.5311, spread 0.0044, max |err| 0.05 %.
* Disk E_2: 18.4069, 18.3890, 18.3968 → mean 18.3976, spread 0.018, max |err| 0.08 %.
* Square E_1: 5.4902, 5.4867, 5.4966 → mean 5.4912, spread 0.010, max |err| 0.15 %.
* Square E_2: 15.1440, **15.2467**, 15.1396. Seeds 0 and 2 converge to the diagonal mode (15.1429; errors +0.01, −0.02 %), seed 1 to the
  midline mode (15.2808; error −0.22 %). The two eigenvalues are 0.9 % apart and share one minimum of the loss curve, so *which* of them
  is returned depends on the initialization; a mean over seeds is meaningless here and the table reports the error to the nearer one.
* The loss-curve vertex is much less reproducible than E_PINN: disk E_2 +0.11/+0.30/+0.34 %, square E_1 +0.49/−0.11/**−1.64** %,
  square E_2 15.16/15.51/15.42.

## R2.2 Convergence for the second eigenvalue of the disk (item 9)

Table (b). E_PINN (network trained at E = 18.5) stays within ±0.08 % for all six configurations (three seeds, the old protocol P0, a 2×64
network, 5000 epochs); over the three nodes 18.0–19.0 it ranges over 18.367–18.408 (−0.14 to +0.08 %). The curve vertex is again the weak
quantity and does **not** improve with effort: +0.11/+0.30/+0.34 % (seeds), +0.22 % (P0), +1.23 % (2×64), **+1.68 % (5000 epochs)** — longer
training lowers the floor on one side of the eigenvalue more than on the other (Λ̄ = 1.09, 0.71, 0.75 at 18.0, 18.5, 19.0). The floor
(≈ 0.85, core fraction 0.4) is ≈ 7 times that of E_1. So the conclusions drawn for E_1 carry over to E_2; the only difference is that P0
happened to be harmless in this one run (single seed).

## R2.3 p = 3 and p = 4: seeds and node dependence (item 9)

Table (c). "Node dependence" = E_PINN of the networks trained at the three central nodes of the 5-node grid.
* **p = 3** (h = 0.75): at the grid minimum (always the node E = 10.0) E_PINN = 9.801, 9.827, 9.814 (errors −0.31, −0.05, −0.18 %;
  max 0.31 %). But E_PINN depends on the node: 9.756–9.911 over nodes 9.25–10.75 and three seeds (−0.77 to +0.81 %), i.e. it drifts by
  ≈ 0.1 per unit of E. The curve vertex is 9.885/9.929/9.917 (+0.55 to +1.00 %).
* **p = 4** (h = 1): the curve is flat (Λ̄ = 4.4–8.0 between E = 14 and 16), the grid minimum is at 16, 15, 15 for the three seeds,
  E_PINN there = 14.895, 14.613, 14.533 (+1.45, −0.47, −1.02 %), node range 14.31–14.90 (−2.6 to +1.5 %), curve vertex +8.2, (+5.6), −0.95 %.
  In addition **2 of the 15 trainings collapsed to u ≡ 0** (mass < 0.5: seed 1 at E = 14, seed 2 at E = 13; the +5.6 % vertex of seed 1 uses
  the collapsed node and is meaningless), as did the 4×256 run of round 1.
* For comparison, p = 2.2: E_PINN over nodes 6.0–7.0 and three seeds is 6.526–6.537 (−0.09 to +0.08 %).
* **Honest conclusion:** the method is validated for p ∈ [2.2, 3] (2.5 in 3-D); at p = 3 the accuracy is ≈ 0.3 % at the grid minimum and
  ≈ 0.8 % if the node is off by one grid step; p = 4 is marginal (1–2.5 %, flat curve, occasional collapse) and should not be presented as
  a success; p = 1.5 fails without regularization (see R2.5). The statement "accuracy degrades from 0.05 % (p = 2.2) to 0.3 % (p = 3) and
  1.5 % (p = 4)" should be replaced by these ranges.
* Referee remark on the h-refinement row (h = 0.125 "+3.7 %"): correct — in that triple the loss is monotone (0.156, 0.116, 0.091), the
  vertex 6.77 lies outside the stencil and is an extrapolation. The rows h = 0.25 and 0.125 are single-seed 3-node stencils; what they show
  is that at h ≤ 0.25 the differences between nodes (0.02–0.07) are of the size of the floor fluctuations, so the curve no longer brackets
  the minimum reliably; the number +3.7 % itself should not be quoted as an error estimate.

## R2.4 Reconciling Remark 1 (wrong loss) and Sec. 5.3(1) (optimization noise) — item 13

Suggested text (3–4 sentences):

> The former values were affected by two independent defects, which should not be confused. The systematic one is the loss: the former
> code solved −Δ_p u = E u with mean(u²) = 1, whose values are 6.935 and ≈ 19.70 for the first two modes of the disk and ≈ 17.6 in the ball
> of R³ (frozen-network values of our re-runs: 6.93, 19.68–19.70, 17.62), against the true 6.532, 18.393 and 14.111; this is what explains
> the former 19.44 (+5.7 %) and 17.45 (+24 %), which are far outside anything optimization noise produces with the correct loss
> (at most 2.3 % in our runs). The random one is the former training protocol (constant learning rate, early stopping), which leaves
> fluctuations of ±0.1–0.2 in the loss curve and makes the located minimum scatter inside a flat basin; this is why the former values
> (6.679, 19.44, 17.45) do not sit exactly on the values of the wrong problem (6.935, 19.70, 17.6) either. With the corrected loss the
> same protocol gives vertex errors of −0.17, +0.47 and +2.26 % over three seeds for E_1 of the disk; that one of these (6.6795) agrees
> with the former 6.679 to three digits is a coincidence — the former number was produced by the former loss (as the original notebook
> shows), and the other two seeds give 6.520 and 6.562.

Accordingly, in Sec. 5.3(1) the phrase "the size of the errors of the former tables" should be weakened to "errors of up to 2 % of either
sign even with the correct loss"; the former *bias* is due to the loss (Remark 1), the noise only adds scatter.
New data behind this: `r2_disk22_legacyE2_P1` (legacy loss, P1, nodes 17–21: loss 8.59, 4.05, 1.66, 1.18, 2.73; frozen-network E =
19.68–19.70 at every node; vertex 19.74).

## R2.5 Regularized p-Laplacian (authors' request)

Loss: ∫(div((|∇u|²+ε²)^{(p−2)/2}∇u) + E|u|^{p−2}u)² + μ(∫|u|^p − 1)², implemented in `core.train_one_reg` (pinn_eigen.py untouched),
P1, coarse grid h = 0.5 on [2.5, 6] then a local 3-node scan h = 0.25 warm-started from the coarse network.

* **Reference** (`refs_reg.py`): radial shooting with the flux r^{d−1}(u'²+ε²)^{(p−2)/2}u'. The regularized problem is **not homogeneous**:
  the value of E for which a positive solution exists depends on the amplitude, hence on the normalization. With ∫|u|^p = 1 on the unit
  disk: p = 1.5: E_ε = 3.9572 (ε = 0.3), 4.0099 (0.1), 4.0170 (0.03), 4.0177 (0.01) → E_1 = 4.0178 (shift −1.51, −0.20, −0.02, −0.00 %);
  p = 4: 15.288 (ε = 0.3), 14.749 (0.1) vs 14.682 (+4.1, +0.46 %). A ±2 % error in the mass changes E_ε by ≤ 0.04 % (p = 1.5).
* **p = 1.5: the regularization cures the failure, for all three ε** (table below). Unregularized: no minimum, Λ̄ = 24–58, E_PINN 5.3–7.8.
  Regularized: a clean minimum at E ≈ 4 with Λ̄_min ≈ 0.10–0.12 (r ≈ 0.32, the level of p = 2.2), mass 1.001, and
  E_PINN = 3.9480 / 3.9988 / 4.0054 for ε = 0.3 / 0.1 / 0.03, i.e. −0.23 / −0.28 / −0.29 % with respect to E_ε and
  **−1.74 / −0.47 / −0.31 % with respect to the true E_1**. Curve vertex (coarse): 3.968 / 4.014 / 4.019 (+0.27 / +0.11 / +0.06 % vs E_ε).
* **Trend as ε → 0:** the model error E_ε − E_1 decreases as predicted (−1.5 → −0.2 → −0.02 %), while the PINN error with respect to E_ε
  stays at ≈ −0.25 % for all ε and the floor rises slightly (0.097 → 0.104 → 0.119). So for ε ≤ 0.1 the total error is dominated by the
  PINN error, not by the regularization, and ε = 0.03 is no better than could be expected from ε = 0.1 beyond the 0.2 % model shift. We did
  not go below ε = 0.03, so we do not know where between 0.03 and the 10⁻¹² safeguard the training breaks down.
* E_PINN depends on the node: 3.943–3.957, 3.992–4.007, 3.999–4.015 over the local nodes 3.75–4.25 (≈ ±0.2 %), and it is below E_ε at the
  grid minimum for every ε. The core fraction is ≈ 0.01, so here the residual is *not* concentrated at the critical point; we have not
  identified the origin of this −0.25 % bias (a candidate is the term |u|^{p−2}u, which is not Lipschitz at u = 0, i.e. at the boundary,
  when p < 2). Single seed.
* **p = 4, ε = 0.1: no improvement** (as expected: for p > 2 the difficulty is the defect at the critical points, not a singularity of the
  operator). Λ̄ = 6.1–8.7 between E = 14 and 17 (unregularized 6.3–8.9), core fraction 0.67–0.84, grid minimum at 16,
  E_PINN = 14.946 (+1.33 % vs E_ε = 14.749, +1.80 % vs E_1), node range 14.38–14.95; essentially identical to the unregularized seed-0 run.
* **Suggested statement for the paper:** for 1 < p < 2 the method works with the regularized operator (tested: p = 1.5, ε ∈ {0.3, 0.1, 0.03},
  unit disk, first eigenvalue, one seed), with a total error of 0.3–0.5 % for ε ≤ 0.1, of which ≤ 0.2 % is the known regularization shift;
  the computed number is an eigenvalue of the regularized problem under the normalization ∫|u|^p = 1, not of the p-Laplacian.

## R2 tables (generated by `analyze_r2.py`)

### Round 2 (a): three seeds, local windows, corrected loss, protocol P1 (p = 2.2)

| case | ref | E_PINN per seed (0,1,2) | mean | spread (max-min) | err of mean % | max abs err % | E_curve per seed | Lam_min per seed |
|---|---|---|---|---|---|---|---|---|
| disk E1 | 6.5317 | 6.5320, 6.5329, 6.5285 | 6.5311 | 0.0044 | -0.01 | 0.05 | 6.543, 6.545, 6.544 | 0.131, 0.130, 0.125 |
| disk E2 | 18.3929 | 18.4069, 18.3890, 18.3968 | 18.3976 | 0.0179 | +0.03 | 0.08 | 18.414, 18.447, 18.456 | 0.832, 0.890, 0.876 |
| square E1 | 5.4950 | 5.4902, 5.4867, 5.4966 | 5.4912 | 0.0100 | -0.07 | 0.15 | 5.522, 5.489, 5.405 | 0.217, 0.235, 0.147 |
| square E2 (two close eigenvalues 15.1429 diag / 15.2808 midline) | nearest | 15.1440→15.1429, 15.2467→15.2808, 15.1396→15.1429 | -- | -- | -- | 0.22 | 15.160, 15.507, 15.420 | 0.197, 0.959, 1.007 |

### Round 2 (b): convergence for the SECOND eigenvalue of the disk, p = 2.2 (ref 18.3929; nodes 18.0, 18.5, 19.0)

| setting | Lam(18.0) | Lam(18.5) | Lam(19.0) | core frac | E_curve | err % | E_PINN (node 18.5) | err % | E_PINN over the 3 nodes | wall min |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline, seed 0 | 1.056 | 0.832 | 1.292 | 0.40 | 18.414 | +0.11 | 18.4069 | +0.08 | 18.3868–18.4069 | 14 |
| baseline, seed 1 | 1.066 | 0.890 | 1.160 | 0.43 | 18.447 | +0.30 | 18.3890 | -0.02 | 18.3857–18.3933 | 13 |
| baseline, seed 2 | 1.144 | 0.876 | 1.259 | 0.41 | 18.456 | +0.34 | 18.3968 | +0.02 | 18.3801–18.3968 | 14 |
| P0 (constant lr, early stop) | 1.103 | 0.937 | 1.226 | 0.43 | 18.433 | +0.22 | 18.3867 | -0.03 | 18.3867–18.4084 | 7 |
| network 2x64 | 1.576 | 1.123 | 1.284 | 0.43 | 18.619 | +1.23 | 18.3823 | -0.06 | 18.3823–18.4007 | 4 |
| 5000 epochs | 1.087 | 0.712 | 0.753 | 0.48 | 18.701 | +1.68 | 18.3857 | -0.04 | 18.3668–18.3904 | 35 |

### Round 2 (c): p = 3 and p = 4, unit disk, E1: three seeds and node dependence of E_PINN

| p | seed | ref | nodes | Lam at nodes | E_PINN at nodes | grid-min node | E_PINN there (err %) | E_curve (err %) | node range of E_PINN, 3 central nodes (err %) |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 | 0 | 9.8315 | [8.5, 9.25, 10.0, 10.75, 11.5] | 3.33, 0.87, 0.60, 1.11, 2.66 | 9.658, 9.775, 9.801, 9.910, 10.001 | 10.0 | 9.8012 (-0.31) | 9.885 (+0.55) | 9.775–9.910 (-0.57 to +0.80) |
| 3.0 | 1 | 9.8315 | [8.5, 9.25, 10.0, 10.75, 11.5] | 2.69, 0.97, 0.57, 1.15, 2.84 | 9.633, 9.758, 9.827, 9.911, 9.981 | 10.0 | 9.8265 (-0.05) | 9.929 (+1.00) | 9.758–9.911 (-0.75 to +0.81) |
| 3.0 | 2 | 9.8315 | [8.5, 9.25, 10.0, 10.75, 11.5] | 3.30, 0.98, 0.67, 1.15, 2.73 | 9.656, 9.756, 9.814, 9.889, 10.000 | 10.0 | 9.8142 (-0.18) | 9.917 (+0.87) | 9.756–9.889 (-0.77 to +0.58) |
| 3.0 | all | | | | | | max abs err 0.31 % | | 9.756–9.911 (-0.77 to +0.81) |
| 4.0 | 0 | 14.6816 | [13.0, 14.0, 15.0, 16.0, 17.0] | 37.46, 8.04, 7.94, 6.26, 8.93 | 13.540, 14.339, 14.454, 14.895, 15.156 | 16.0 | 14.8952 (+1.45) | 15.886 (+8.20) | 14.339–14.895 (-2.34 to +1.45) |
| 4.0 | 1 | 14.6816 | [13.0, 14.0, 15.0, 16.0, 17.0] | 12.53, 1344014.72, 4.44, 5.21, 8.99 | 13.839, 406.516, 14.613, 14.900, 15.133 | 15.0 | 14.6130 (-0.47) | 15.500 (+5.57) | 14.613–14.900 (-0.47 to +1.48); COLLAPSED to u=0 at E=[14.0] |
| 4.0 | 2 | 14.6816 | [13.0, 14.0, 15.0, 16.0, 17.0] | 16126.52, 6.12, 6.11, 6.54, 8.35 | 49.069, 14.306, 14.533, 14.801, 15.191 | 15.0 | 14.5326 (-1.02) | 14.543 (-0.95) | 14.306–14.801 (-2.56 to +0.81); COLLAPSED to u=0 at E=[13.0] |
| 4.0 | all | | | | | | max abs err 1.45 % | | 14.306–14.900 (-2.56 to +1.48) |

### Round 2 (d): legacy loss near the published second value of the disk (19.44; true 18.393)

nodes [17.0, 18.0, 19.0, 20.0, 21.0], legacy loss [8.591, 4.054, 1.664, 1.175, 2.732], frozen-network E [19.675, 19.685, 19.7, 19.704, 19.702]; grid minimum 20.0, 3-pt vertex 19.739.

### Regularised p-Laplacian, unit disk: div((|grad u|^2+eps^2)^{(p-2)/2} grad u) + E|u|^{p-2}u, normalisation int|u|^p = 1

| p | eps | E_eps (shooting, mass 1) | E1 (eps=0) | coarse 3-pt vertex | local 3-pt vertex | E_PINN (grid-min node) | err vs E_eps % | err vs E1 % | E_PINN over local nodes | Lam_min | core frac | mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 0 (1e-12 safeguard) | -- | 4.0178 | no minimum (Lam 24–58) | -- | 5.289 | -- | +31.63 | 5.29–7.84 | 24.2 | 0.36 | 0.955 |
| 1.5 | 0.3 | 3.9572 | 4.0178 | 3.968 (+0.27 vs E_eps) | 3.987 | 3.9480 (node 4.0) | -0.23 | -1.74 | 3.943–3.957 | 0.097 | 0.01 | 1.001 |
| 1.5 | 0.1 | 4.0099 | 4.0178 | 4.014 (+0.11 vs E_eps) | 4.020 | 3.9988 (node 4.0) | -0.28 | -0.47 | 3.992–4.007 | 0.104 | 0.00 | 1.001 |
| 1.5 | 0.03 | 4.0170 | 4.0178 | 4.019 (+0.06 vs E_eps) | 4.018 | 4.0054 (node 4.0) | -0.29 | -0.31 | 3.999–4.015 | 0.119 | 0.01 | 1.001 |
| 4.0 | 0.1 | 14.7490 | 14.6816 | 15.918 (+7.93 vs E_eps) | -- | 14.9459 (node 16.0) | +1.33 | +1.80 | 13.603–15.206 | 6.097 | 0.67 | 0.978 |
