#!/bin/bash
# waits for seed-0 job, then runs seeds 1 and 2 (same windows). 1 core.
while pgrep -f run_pinn_all.sh >/dev/null; do sleep 30; done
cd /home/user/PINN-eigenvalues-v2
experiments/fem/run_pinn_all.sh 1 > experiments/fem/logs/pinn_s1.log 2>&1
experiments/fem/run_pinn_all.sh 2 > experiments/fem/logs/pinn_s2.log 2>&1
