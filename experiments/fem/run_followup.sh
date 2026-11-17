#!/bin/bash
# waits for seed-0 job, then seed 1 (all windows) and seed 2 (four-well and harmonic only, cheaper). 1 core.
while pgrep -f run_pinn_all.sh >/dev/null; do sleep 30; done
cd /home/user/PINN-eigenvalues-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
experiments/fem/run_pinn_all.sh 1 > experiments/fem/logs/pinn_s1.log 2>&1
python -m experiments.fem.pinn_windows four_well --centers 9.85 20.19 33.23 39.47 --seed 2 > experiments/fem/logs/pinn_s2.log 2>&1
python -m experiments.fem.pinn_windows harm_w10 --centers 14.28 29.13 45.7 62.58 --seed 2 >> experiments/fem/logs/pinn_s2.log 2>&1
