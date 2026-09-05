# PINN-eigenvalues-v2

Refactored code for the paper:

**"A PINNs approach for the computation of eigenvalues in elliptic problems"**  
Julián Fernández Bonder & Ariel Salort  
[arXiv:2507.03126](https://arxiv.org/abs/2507.03126)

## Structure

- `pinn_eigen.py` — core module with all reusable components
- `FBS_PINN_notebook_v2.ipynb` — clean notebook reproducing all 9 examples from the paper

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
