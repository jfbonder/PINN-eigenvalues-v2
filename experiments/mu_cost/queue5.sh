#!/bin/bash
cd /home/user/PINN-eigenvalues-v2; export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P="python -u -m experiments.mu_cost"
echo "=== $(date +%T) resume broad scan s0"
$P.scan --tag mucost_scan_broad_mu450_s0 --E 15:105:5 --mu 450 --seed 0
echo "=== $(date +%T) narrow mu=5000"
$P.scan --tag mucost_t1_narrow5000 --E 95,96.75,98.5,100.25,102 --mu 5000 --seed 0 --epochs 5000 --reset 25
echo "=== $(date +%T) RQ patience 3000"
$P.rayleigh_baseline --seed 0 --K 6 --epochs 3000 --patience 3000
echo "=== $(date +%T) DONE5"
