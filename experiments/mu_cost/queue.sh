#!/bin/bash
# Sequential, resumable job queue (1 core). Each step skips already-finished points.
cd /home/user/PINN-eigenvalues-v2
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P="python -u -m experiments.mu_cost"
log() { echo "=== $(date +%T) $*"; }
log "item5 RQ seed0";        $P.rayleigh_baseline --seed 0 --K 6 --epochs 3000
log "item5/2 broad scan s0"; $P.scan --tag mucost_scan_broad_mu450_s0 --E 15:105:5 --mu 450 --seed 0
log "item4 rows"
$P.scan --tag mucost_t1_11 --E 15.5,17.5,19.5,21.5,23.5 --mu 3000 --seed 0
$P.scan --tag mucost_t1_12 --E 44.5,46.5,48.5,50.5,52.5 --mu 3000 --seed 0
$P.scan --tag mucost_t1_22 --E 74.5,76.5,78.5,80.5,82.5 --mu 3000 --seed 0
$P.scan --tag mucost_t1_13 --E 94.5,96.5,98.5,100.5,102.5 --mu 3000 --seed 0
log "item1 mu sweep (reduced net 3x64, 3000 pts, 2000 ep)"
for s in 0 1; do for mu in 1000 100 10 1 30 60 113 10000 300; do
  $P.scan --tag mucost_mu${mu}_s${s} --E 40:60:2 --mu $mu --seed $s --hidden 64 --epochs 2000 --npts 3000
done; done
log "item1 full-settings check mu=1000 (3x128, 5000 pts, 3000 ep)"
$P.scan --tag mucost_mu1000_full_s0 --E 40:60:2 --mu 1000 --seed 0
log "item2 seeds 1,2"
for s in 1 2; do $P.scan --tag mucost_scan_broad_mu450_s$s --E 15:55:5 --mu 450 --seed $s; done
log "RQ seed 1";  $P.rayleigh_baseline --seed 1 --K 6 --epochs 3000
log "DONE"
