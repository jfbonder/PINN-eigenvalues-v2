#!/bin/bash
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P="python -u -m experiments.mu_cost"
OFFB=$(python -c "import numpy as np;print(','.join('%.2f'%x for x in np.arange(17.3,105,5)))")
for s in 0 1 2; do
 echo "=== $(date +%T) (b) offset scan seed $s load $(cut -d' ' -f1 /proc/loadavg)"
 $P.scan --tag mucost_scan_broad_off_s$s --E $OFFB --mu 450 --seed $s
done
for s in 0 1 2; do
 echo "=== $(date +%T) (b) RQ seed $s ES"; $P.rayleigh_baseline --seed $s --K 6 --epochs 3000 --tag mucost_r2_rq_s${s}_es
 echo "=== $(date +%T) (b) RQ seed $s noES"; $P.rayleigh_baseline --seed $s --K 6 --epochs 3000 --patience 3000 --tag mucost_r2_rq_s${s}_noes
done
OFF=$(python -c "import numpy as np;print(','.join('%.2f'%x for x in np.arange(15.6,55,2.5)))")
echo "=== $(date +%T) (a) mu offset seed1"
for mu in 30 75 150 219 300 1000; do $P.scan --tag mucost_off_mu${mu}_s1 --E $OFF --mu $mu --seed 1 --hidden 64 --epochs 2000 --npts 3000; done
echo "=== $(date +%T) DONEB"
