#!/bin/bash
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P="python -u -m experiments.mu_cost"
OFF=$(python -c "import numpy as np;print(','.join('%.2f'%x for x in np.arange(15.6,55,2.5)))")
echo "=== $(date +%T) (a) mu offset seed0 load $(cut -d' ' -f1 /proc/loadavg)"
for mu in 30 75 150 219 300 1000; do $P.scan --tag mucost_off_mu${mu}_s0 --E $OFF --mu $mu --seed 0 --hidden 64 --epochs 2000 --npts 3000; done
echo "=== $(date +%T) (c) cold start 10pi^2"
for s in 0 1 2; do for mu in 5000 20000; do
 $P.scan --tag mucost_t1_cold_mu${mu}_s${s} --E 95,96.75,98.5,100.25,102 --mu $mu --seed $s --epochs 5000 --reset 25
done; done
echo "=== $(date +%T) DONEA"
