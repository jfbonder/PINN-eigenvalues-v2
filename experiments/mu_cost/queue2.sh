#!/bin/bash
# runs after queue.sh
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f experiments/mu_cost/queue.sh >/dev/null; do sleep 30; done
P="python -u -m experiments.mu_cost"
echo "=== $(date +%T) paper-protocol sweep (notebook Ex.2: linspace(0,105,80), mu=500, 5000 ep, reset 25)"
$P.scan --tag mucost_t1_sweep --E $(python -c "import numpy as np;print(','.join('%.6f'%x for x in np.linspace(0,105,80)))") --mu 500 --seed 0 --epochs 5000 --reset 25
echo "=== $(date +%T) broad scan s0 (h=5)"
$P.scan --tag mucost_scan_broad_mu450_s0 --E 15:105:5 --mu 450 --seed 0
echo "=== $(date +%T) RQ patience 3000"
$P.rayleigh_baseline --seed 0 --K 6 --epochs 3000 --patience 3000
echo "=== $(date +%T) DONE2"
