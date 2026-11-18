# PINN-eigenvalues-v2

Refactored code for the paper:

**"A PINNs approach for the computation of eigenvalues in elliptic problems"**  
Julián Fernández Bonder & Ariel Salort  
[arXiv:2507.03126](https://arxiv.org/abs/2507.03126)

## Structure

| path | content |
|---|---|
| `pinn_eigen.py` | core module with all reusable components (see below) |
| `examples/` | every example of the current version of the paper (`paper/v4`), with the settings stated there: `configs.py` (one entry per figure/table), `run.py` (command line), `core.py` (driver), `references.py` (exact/reference eigenvalues), `FBS_PINN_examples.ipynb` (notebook, one section per section of the paper) |
| `paper/v3/`, `paper/v4/` | LaTeX source of the previous arXiv version (unchanged) and of the revised manuscript (changes in blue) |
| `experiments/` | all the numerical experiments of the revised manuscript, with their raw results (see below) |
| `theory/` | notes, LaTeX fragments and checks used for the analysis of the method (stability, choice of the penalty, p-Laplacian) |
| `audit/` | consistency check of the previous version (text vs. code vs. tables), with the scripts used |

## Experiments

Every experiment has its own folder under `experiments/` with a `README.md` that gives the exact settings, the commands to re-run it and the
results. Raw outputs (one JSON record per value of E, with seeds, timings and Monte Carlo standard errors) are in `experiments/results/`;
the tables and figures of the paper are generated from these files by the `analyze*`/`make_*` scripts. Everything was run on CPU
(1 thread per process); `experiments/ENVIRONMENT.md` records the software versions and the measured cost per epoch.

| folder | question | paper |
|---|---|---|
| `mu_cost/` | Sensitivity of the eigenvalues and of the loss curve to the penalty parameter μ (collapse and spurious-minimum thresholds); seeds and spurious minima; refinement of the minima; re-run of the square example; cost of the E-scan compared with a Rayleigh-quotient PINN with deflation | Sections 3, 6.1–6.3 |
| `fem/` | Independent finite-element references (scikit-fem, curved P2 elements, Richardson extrapolation) for the double-well and four-well potentials in the unit ball of R³, the harmonic potential in the disk, the annulus, the square and the triangle; PINN runs on the same problems; Monte Carlo standard errors; blind scan of the double-well loss curve; ball of R⁴ | Sections 4.2–4.5 |
| `highdim/` | Unit ball in R⁴ (full warm-started scan, `scan_r4.py`, figure by `plot_scan_r4.py`) and in R¹⁰ (`run_d10.py`) | Section 4.2 |
| `plap/` | p-Laplacian: reference eigenvalues by radial shooting (`refs.py`) and P1-FEM (`fem_plap.py`); PINN results for the disk, the square and the ball of R³ with the homogeneous loss; convergence in network size, number of points, epochs and grid step; seeds; p = 3, 4; regularized p-Laplacian | Section 5, Appendix C |
| `smoke/` | quick checks of the module and the benchmark of the cost per epoch | – |
| `plap35/` | *Exploratory, not in the paper.* p = 3.5 in the disk: sources of the oscillation of the loss and remedies (singular input feature, mixed formulation) | – |
| `cross/` | *Exploratory, not in the paper.* Non-convex domains (cross, L-shape): hard vs. weak boundary conditions, FEM references with graded meshes | – |
| `report_L_p35/` | *Exploratory, not in the paper.* Report (PDF) summarizing the L-shape and p = 3.5 studies | – |

Two corrections with respect to the earlier code, documented in the folders above:

* **p-Laplacian loss.** The earlier code used the residual Δ_p u + E u with the normalization ∫u² = 1, which is not homogeneous, so the computed
  value depended on the normalization. `loss_fn` now uses Δ_p u + E|u|^{p−2}u with ∫|u|^p = 1, as stated in the paper; the old behaviour is
  available with `legacy_plap=True` (see `experiments/ORIGINAL_CODE_COMPARISON.md` and `experiments/plap/README.md`).
* **Sampler in the ball of R⁴.** `sample_ball_4d` used a hyperspherical-angle parametrization with wrong angular densities, so the points
  were not uniform. It now samples a Gaussian direction times a radius U^{1/4}. The R⁴ results were recomputed; the old ones are kept with the
  suffix `_OLDSAMPLER`.

Typical commands (from the repository root):

```bash
python -m experiments.plap.refs                       # p-Laplacian references by shooting
python -m experiments.fem.run_fem double_well four_well   # FEM references
python -m experiments.fem.pinn_round2 ball4 --seed 0  # PINN in the ball of R^4
python -m experiments.highdim.scan_r4                 # full scan in R^4
```

Dependencies for the experiments: `torch numpy scipy matplotlib scikit-fem meshio pandas`.

## What's in `pinn_eigen.py`

| Component | Description |
|---|---|
| `EigenFunctionNN` | Feedforward network with automatic BC enforcement via boundary factor B(x) |
| `loss_fn` | PINN loss for linear (Laplacian) and nonlinear (p-Laplacian) problems |
| `train_one` | Train for a single E value with early stopping |
| `sweep` | Scan a range of E values with warm initialization |
| `refine` | Detect and locally refine minima of the loss curve (re-sampled scalar minimization, earlier version) |
| `local_minima`, `a_posteriori` | Detection of the minima; closed-form refinement (Rayleigh quotient of the frozen network, or E_PINN for p ≠ 2) with Monte Carlo standard errors, r, η and the Weinstein enclosure |
| `kato_temple`, `temple_bracket`, `cosine_lr` | Kato–Temple bound, Temple bracket for E₁, learning-rate decay of the p-Laplacian protocol |
| `sample_disk`, `sample_square`, `sample_triangle`, `sample_ball_3d`, `sample_ball_4d`, `sample_ball`, `sample_annulus` | Domain samplers (`sample_ball(n, dim)`: any dimension) |
| `boundary_disk`, `boundary_square`, `boundary_triangle`, `boundary_annulus`, `boundary_ball` | Boundary factor functions |
| `V_zero`, `V_harmonic`, `V_double_well_gaussian`, `V_four_well_bump` | Potential functions |
| `plot_loss_curve`, `plot_eigenfunction_disk`, `plot_eigenfunction_square`, `plot_eigenfunction_triangle` | Plotting utilities |

## Examples of the paper

The `examples/` package reproduces every example of the paper (`paper/v4/FBS-v4.tex`). Each example is one entry
of `examples/configs.py` with the settings stated in the paper (domain, boundary factor, potential, p, μ, window,
grid step, epochs, points, type of start, threshold, reference eigenvalues, figure/table it corresponds to). A run
trains along the E-grid, detects the minima of the loss curve, refines each one with the closed-form Rayleigh
quotient of the frozen network and computes the a posteriori quantities of Section 3.1 (residual norm r, η,
Weinstein enclosure, Kato–Temple and Temple bounds with the gap taken from the reference spectrum); for the
p-Laplacian it runs the local three-node scans of Section 5 and reports E_PINN and E_curve. It writes a JSON file,
the loss curve and the computed eigenfunctions, and prints a comparison with the exact/reference values and with
the values reported in the paper.

```bash
python -m examples.run --list                      # list the examples
python -m examples.run disk --quick                # smoke test: tiny settings, a few seconds
python -m examples.run disk                        # settings of the paper
python -m examples.run ball4_windows --windows 0   # only the first window of an example
python -m examples.run all --quick                 # every example, smoke test (about 2 minutes)
python -m examples.run square --seed 1 --threads 2 --out my_runs/ --mu 1000   # overrides
```

Output goes to `examples_out/` (`<name>_s<seed>.json`, `_loss.png`, `_eigenfunctions.png`).

| example | paper | example | paper |
|---|---|---|---|
| `disk` | Fig. lossNN (Sec. 3) | `annulus` | Fig. loss.lap.annulus |
| `square_44_55` | Fig. eigen55 | `harmonic_w1`, `harmonic_w10` | Figs. pot2, pot1 |
| `square` | Table 1 | `double_well`, `four_well` | Fig. well-two / well-four |
| `ball3`, `ball4` | Fig. loss3D | `double_well_windows`, `four_well_windows`, `harmonic_w10_windows` | Table fem, Fig. fem |
| `ball3_windows`, `ball4_windows` | Table 2 | `double_well_blind` | Table blind |
| `ball10` | Sec. 4.2 (ball of R¹⁰) | `plap_disk`, `plap_square`, `plap_ball3` | Table plap, Fig. plap |
| `triangle`, `triangle_windows` | Fig. loss.lap.triang, Table 4 | `plap_disk_p3`, `plap_disk_p4` | Table plap.p |

The notebook `examples/FBS_PINN_examples.ipynb` walks through the same examples, one section per section of the
paper; it runs in quick mode by default (a few minutes in total) and with the paper's settings when
`PAPER_SETTINGS = True`. Its last section shows how to modify an example (`dataclasses.replace`) and how to set up a
new domain (sampler + boundary factor). Full runs take from minutes (three-node windows) to hours (full scans) on
one CPU thread; the cost of each experiment is recorded in `experiments/`.

Using the module directly:

```python
import numpy as np, pinn_eigen as pe
from functools import partial

# Laplacian eigenvalues in the unit disk
sampler  = partial(pe.sample_disk, radius=1.0)
model_fn = lambda: pe.EigenFunctionNN(dim=2, boundary_fn=pe.boundary_disk)
E_values = np.arange(0.0, 32.01, 0.5)

residuals, models = pe.sweep(model_fn, E_values, sampler, pe.V_zero, mu=100.0, reset_every=10**9)
for i in pe.local_minima(residuals, threshold=1.0):
    ev = pe.a_posteriori(models[i], sampler, pe.V_zero, E_values[i])
    print(ev["rho_res"], ev["weinstein"])     # refined eigenvalue and Weinstein enclosure
```

## Requirements

```
torch
numpy
scipy
matplotlib
```

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{FernandezBonderSalort2025PINN,
  author        = {Fern{\'a}ndez Bonder, Juli{\'a}n and Salort, Ariel},
  title         = {A {PINNs} approach for the computation of eigenvalues in elliptic problems},
  year          = {2025},
  eprint        = {2507.03126},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2507.03126}
}
```

GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
