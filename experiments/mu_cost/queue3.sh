#!/bin/bash
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f "experiments/mu_cost/queue2.sh" >/dev/null; do sleep 30; done
P="python -u -m experiments.mu_cost"
echo "=== $(date +%T) narrow window [95,102], cold start, mu=450, 5000 ep"
$P.scan --tag mucost_t1_narrow --E 95,96.75,98.5,100.25,102 --mu 450 --seed 0 --epochs 5000 --reset 25
echo "=== $(date +%T) DONE3"
