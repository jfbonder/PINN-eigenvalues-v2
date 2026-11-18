#!/bin/bash
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f r2_qA.sh >/dev/null; do sleep 20; done
P="python -u -m experiments.mu_cost"
OFF=$(python -c "import numpy as np;print(','.join('%.2f'%x for x in np.arange(15.6,55,2.5)))")
for mu in 1000 300 219; do $P.scan --tag mucost_off_mu${mu}_s1 --E $OFF --mu $mu --seed 1 --hidden 64 --epochs 2000 --npts 3000; done
echo "=== DONEA2"
