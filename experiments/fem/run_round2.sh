#!/bin/bash
# resumable, 1 core, sequential
cd /home/user/PINN-eigenvalues-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for t in ball3 scan ball4; do
  python -m experiments.fem.pinn_round2 $t --seed 0 >> experiments/fem/logs/round2_$t.log 2>&1
done
