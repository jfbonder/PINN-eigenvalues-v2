#!/bin/bash
cd /home/user/PINN-eigenvalues-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.fem.pinn_windows harm_w10 --centers 62.58 --seed 1 >> experiments/fem/logs/pinn_s1.log 2>&1
python -m experiments.fem.pinn_windows four_well --centers 9.85 20.19 33.23 39.47 --seed 2 >> experiments/fem/logs/pinn_s2.log 2>&1
python -m experiments.fem.pinn_windows harm_w10 --centers 14.28 29.13 45.7 62.58 --seed 2 >> experiments/fem/logs/pinn_s2.log 2>&1
