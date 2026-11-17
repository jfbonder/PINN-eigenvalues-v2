#!/bin/bash
# runs after queue.sh
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f experiments/mu_cost/queue.sh >/dev/null; do sleep 30; done
P="python -u -m experiments.mu_cost"
echo "=== $(date +%T) broad scan s0 (with refinement outputs)"
$P.scan --tag mucost_scan_broad_mu450_s0 --E 15:105:5 --mu 450 --seed 0
echo "=== $(date +%T) RQ patience 3000"
$P.rayleigh_baseline --seed 0 --K 6 --epochs 3000 --patience 3000
echo "=== $(date +%T) DONE2"
