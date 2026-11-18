# PINN-eigenvalues-v2

Refactored code for the paper:

**"A PINNs approach for the computation of eigenvalues in elliptic problems"**  
Julián Fernández Bonder & Ariel Salort  
[arXiv:2507.03126](https://arxiv.org/abs/2507.03126)

## Structure

| path | content |
|---|---|
| `pinn_eigen.py` | core module with all reusable components (see below) |
| `FBS_PINN_notebook_v2.ipynb` | notebook reproducing the examples of the first version of the paper |
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
| `refine` | Detect and locally refine minima of the loss curve |
| `sample_disk`, `sample_square`, `sample_triangle`, `sample_ball_3d`, `sample_ball_4d`, `sample_annulus` | Domain samplers |
| `boundary_disk`, `boundary_square`, `boundary_triangle`, `boundary_annulus`, `boundary_ball` | Boundary factor functions |
| `V_zero`, `V_harmonic`, `V_double_well_gaussian`, `V_four_well_bump` | Potential functions |
| `plot_loss_curve`, `plot_eigenfunction_disk`, `plot_eigenfunction_square`, `plot_eigenfunction_triangle` | Plotting utilities |

## Usage (Google Colab)

Upload `pinn_eigen.py` alongside the notebook, or add to your Colab session:

```python
import pinn_eigen as pe
from functools import partial
import numpy as np

# Example: Laplacian eigenvalues on the unit disk
sampler  = partial(pe.sample_disk, radius=1.0)
V_fn     = pe.V_zero
model_fn = lambda: pe.EigenFunctionNN(dim=2, boundary_fn=pe.boundary_disk)

E_values = np.linspace(0, 30, 60)

residuals, models = pe.sweep(model_fn, E_values, sampler, V_fn, mu=100.0)
pe.plot_loss_curve(E_values, residuals)
eigenvalues, idx = pe.refine(models, E_values, residuals, sampler, V_fn, threshold=0.5)
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
