#!/bin/bash
# sequential, resumable (per-E records are saved; re-running skips finished E points). 1 core.
cd /home/user/PINN-eigenvalues-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=${1:-0}
python -m experiments.fem.pinn_windows double_well --centers 9.58 19.86 32.9 39.2 --seed $S
python -m experiments.fem.pinn_windows four_well --centers 9.85 20.19 33.23 39.47 --seed $S
python -m experiments.fem.pinn_windows harm_w10 --centers 14.28 29.13 45.7 62.58 --seed $S
