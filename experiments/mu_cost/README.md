# mu_cost: mu-sensitivity, seeds/spurious minima, refinement, Table-1 re-run, cost vs Rayleigh-PINN

Unit square [0,1]^2, V=0, Dirichlet, B(x)=x(1-x)y(1-y), exact eigenvalues pi^2(m^2+n^2). Integrals are sample means (|Omega|=1, so
mean = integral). Everything on **1 CPU thread** (torch.set_num_threads(1)); the machine was shared with other agents (load ~4 on 4 cores), so
wall times are indicative (they include some contention), but all methods ran under the same conditions. Container restarted once at ~01:18; the
last runs were resumed from checkpoints (resumable JSON per node).

Code: `lib.py` (helpers, resumable warm-start chain), `scan.py` (CLI), `rayleigh_baseline.py` (item 5), `analyze_mu.py`, `analyze_scan.py`,
`make_report.py`, `queue*.sh` (job queues). Results: `experiments/results/mucost_*.json`, `mu_summary.json`, `report_data.json`; logs in `logs/`.
Figures in `figures/`; LaTeX tables in `tables.tex`.

Common settings ("paper settings"): MLP 3x128 tanh, Adam lr 1e-3, 5000 fresh uniform points per epoch, early stopping tol 1e-5 / patience 300,
warm start along E (network of previous E reused), seeds recorded in the JSON. Every node is evaluated on a *fresh* 20000-point sample:
Residual = mean(res^2), mass = mean u^2, r = sqrt(mean res^2 / mean u^2). Eigenvalue refinements: see item 3.
Certificate checks use 50000 fresh points at the reported E.

## 1. mu sensitivity (R1.3) - window [40,60], 5pi^2=49.348 (double)
Grid E=40,42,...,60 (11 nodes, warm start), 2 seeds, mu in {1,10,30,60,100,113,300,1000,10000}. D = max dist(E,sigma) over the window =
10.65, **D^2 = 113.5**; theory: mu >= D^2 faithful, D^2/2=57 <= mu < D^2 spurious, mu < D^2/2 collapse (u=0 when d^2 >= 2mu, norm 1-d^2/2mu).
**Reduced network 3x64, 3000 points, 2000 epochs max** (cost). Validation of reduced vs paper settings at mu=1000 (full 3x128/5000 pts/3000 ep, seed 0,
`mucost_mu1000_full_s0`): normalised quantities agree: loss_res at E=48/50/52 = 1.98/0.57/7.75 (full) vs 2.02/0.49/7.13 (reduced); mass within 0.03;
closed-form Rayleigh quotient at every node 49.343-49.353 (full) vs 49.346-49.350 (reduced) (node E=40 reduced: 49.057, under-trained). Full took 1071 s, reduced 305 s.

| mu | mu/D^2 | observed collapsed nodes (mass<0.1) / predicted | detected minimum (recorded curve) | eigenvalue estimate (closed-form RQ), rel. err | comment |
|---|---|---|---|---|---|
| 1 | 0.009 | 11 / 10 | spurious minima (44,54,58 / 50,56) | none (u=0) | all collapsed; warm-started u=0 is absorbing, even where theory says u!=0 |
| 10 | 0.09 | 11 / 7 | spurious (42,46,58 / 48) | none | same |
| 30 | 0.26 | 5 / 3 | 42 (spurious) and 50 | 49.3465/49.3469 (<1e-4) | collapse at far nodes; spurious zero of the recorded residual at 42 |
| 60 | 0.53 | 1-2 / 1 | 50 | 49.3474/49.3475 | collapse only at window end |
| 100 | 0.88 | 0 / 0 | 50 | 49.3476/49.3484 | min norm 0.42 as predicted (1-d^2/2mu = 0.43) |
| 113 | 1.0 | 0 / 0 | 50 | 49.3479/49.3481 | |
| 300 | 2.6 | 0 / 0 | 50 | 49.3478/49.3466 | |
| 1000 | 8.8 | 0 / 0 | 50 | 49.3485/49.3485 | |
| 10000 | 88 | 0 / 0 | 50 | 49.358/49.3483 | curve badly distorted (stiff penalty, not converged in 2000 epochs; MC floor of mu*(mass-1)^2), minimum location still right |

Conclusion: eigenvalue location is flat in mu over [D^2, 10^4] (|error| < 2e-4 relative); the curve follows d^2-d^4/(4mu) for mu in [60,1000]
(figure `mu_loss_curves.png`; norm vs prediction in `mu_mass.png`, RMS deviation of mean u^2 from 1-d^2/2mu: ~0.01 for mu>=100); thresholds D^2/2 (collapse)
and D^2 (spurious minima of the *recorded residual*) confirmed (mu=30 spurious zero at E=42; the normalised quotient r^2 shows a single minimum at 50 for all mu >= 30). For
cold starts a different (larger) lower bound mu >~ E^2/2 is needed (item 4/5). Optimisation hardness grows for very large mu. Data: `mu_summary.json`.

## 2. Seeds / spurious minima (R1.6) - window [15,55], step 5, mu=450 (=2D^2 rule, D^2=219), paper settings, seeds 0,1,2
Detected minima (recorded and normalised curve): 20 and 50 for all three seeds, no spurious minimum. Refined (closed-form RQ on 300^2 grid):

| seed | E_hat(2pi^2) | err | r (cert.) | KT bound r^2/gap | E_hat(5pi^2) | err | r | KT |
|---|---|---|---|---|---|---|---|---|
| 0 | 19.7394 | 2.3e-4 | 0.158 | 8e-4 | 49.3486 | 6.2e-4 | 0.457 | 7.0e-3 |
| 1 | 19.7397 | 4.5e-4 | 0.221 | 1.6e-3 | 49.3485 | 4.7e-4 | 0.348 | 4.1e-3 |
| 2 | 19.7450 | 5.8e-3 | 0.678 | 1.6e-2 | 49.3474 | 6.4e-4 | 0.338 | 3.9e-3 |

Spread of E_hat over seeds: 5.6e-3 (2pi^2; seed 2 under-trained) and 1.2e-3 (5pi^2). Weinstein |E-E_k| <= r holds in all cases (error is 2-3 orders below r);
Kato-Temple error <= r^2/gap holds in all cases (the bound is also 5-25x larger than the actual error). Seed overlay: `figures/seeds_overlay.png`.
**Spurious minima do occur from optimisation, not from the loss**: (a) in the paper-protocol sweep (item 4) after a cold re-initialisation (reset_every=25 at E=99.7) the network
collapsed to u=0 for E=99.7..105: recorded residual 1e-5 (the minimum of the recorded curve, would be picked by any threshold!), mass=0.000, but r=30-40;
(b) mu too small (item 1). Both are flagged by mass << 1 and by the certificate r (huge), and are invisible in the residual alone. Recommend recording mass and r.

## 3. Refinement comparison (16 minima from all chains; 14 non-collapsed)
Absolute error |E_hat - E_exact| on non-collapsed minima (chains: paper sweep, 2 cold rows, broad scans 3 seeds, narrow mu=5000):

| estimator | median | mean | max |
|---|---|---|---|
| existing refine(): Brent on re-sampled 10^4-pt objective | 6.6e-3 | 1.7e-2 | 1.3e-1 |
| closed-form RQ, -sum(u Lu)/sum u^2, fixed 10^5 MC sample | 1.1e-3 | 1.4e-3 | 5.5e-3 |
| closed-form RQ, same on 300x300 midpoint grid | **6.6e-4** | 1.4e-3 | 5.8e-3 |
| gradient form sum|grad u|^2/sum u^2, grid | 6.7e-4 | 1.4e-3 | 5.8e-3 |
| 3-point parabola of r^2 (node spacing 1.3-5) | 9.7e-3 | 9.1e-2 | 1.0 |
| 3-point parabola of recorded residual | 1.1e-2 | 5.4e-2 | 0.43 |

The closed-form Rayleigh quotient on a fixed sample/grid is the most accurate (10x better median, 20x better worst case than refine(); e.g. 5pi^2 cold row: 4e-3 vs 0.126),
free of noise and costs ~10 s per node (grid). Its residual error is the network's own (Kato-Temple, quadratic in r). Lap and gradient forms coincide. The parabola
is *not* reliable at these coarse spacings (loss values at different nodes carry different training floors, so the vertex is biased by up to ~h/2). Figure `refine_compare.png`.

## 4. Table 1 re-run (paper-protocol notebook Ex. 2: linspace(0,105,80), mu=500, 5000 epochs, early stop 300/1e-5, reset_every=25, seed 0)
Wall time 6534 s training + 413 s refinement (80 nodes, avg 82 s/node). Reported E = closed-form RQ (grid). Residual = fresh-sample mean res^2 *at the reported E*, r = its Weinstein radius.

| (m,n) | exact | computed E | rel. err | Residual | r | Weinstein |E-E_k|<=r | old refine() value (err) |
|---|---|---|---|---|---|---|---|
| (1,1) | 19.7392 | 19.739909 | 3.5e-5 | 9.7e-2 | 0.31 | yes | 19.7378 (1.4e-3) |
| (1,2)/(2,1) | 49.3480 | 49.348167 | 3.0e-6 | 7.3e-2 | 0.27 | yes | 49.3336 (1.4e-2) |
| (2,2) | 78.9568 | 78.956161 | 9.0e-6 | 2.0e-1 | 0.44 | yes | 78.9820 (2.5e-2) |
| (1,3)/(3,1) | 98.6960 | 98.695973 | 1.0e-6 | 9.3e-2 | 0.30 | yes | 98.6756 (2.0e-2) |

The v3 row (1,3): E=98.594, Residual 1.4e-3 violated Weinstein (r=0.037 < 0.10); in our run the (1,3) value has Residual 9.3e-2 (r=0.30>=7e-5 actual error). Residuals in this
protocol are 0.07-0.2 (the residual of the *trained* net, not Lambda=dist^2 which would be ~1e-9). The multiple eigenvalues 5pi^2, 10pi^2 appear as one minimum (the scan
with one network does not resolve multiplicity). Extra (non-eigenvalue) minima: node E=102.34 (and the recorded-curve minimum at 99.7) are collapse artefacts (mass 0, r=30), see item 2.
Cold chains (3 start points, mu=3000, 5 nodes each, step 2, no warm-up): (1,1) E=19.7416 (err 2.4e-3), (1,2) 49.3440 (4.0e-3) fine, but (2,2) at E=74.5.. and (1,3) at E=94.5.. **failed**
(mass 0.55-0.9, Residual 600-1400): cold start at large E does not work with mu=3000; warm-start continuation from lower E is essential (also: mu=450 cold start at E=95..102 collapsed completely,
`mucost_t1_narrow`; mu=5000 > E^2/2 worked, see item 5). This is the cold-start threshold mu >~ E^2/2 of the theory notes.

## 5. Cost vs Rayleigh-quotient PINN with deflation (R1.1) - 1 thread
Baseline: same MLP and B(x); loss = mean|grad u|^2/mean u^2 + beta sum_{j<k} (mean u u_j)^2/(mean u^2 mean u_j^2), beta=200 (> max gap), Adam lr 1e-3, 5000 fresh
points/epoch, <=3000 epochs, early stopping 300/1e-5 (also a no-early-stop variant, patience 3000); eigenvalue = gradient-form quotient on the 300^2 grid. K=6 (19.74, 49.35 x2, 78.96, 98.70 x2).
Per-eigenvalue results (rel. error vs the exact value nearest the computed one):

| k | exact | RQ seed0 (ES): E, err, s | RQ seed1 (ES): err, s | RQ seed0 no-ES: E, err, s |
|---|---|---|---|---|
| 1 | 19.739 | 19.761, 1.1e-3, 32 | 6.9e-4, 25 | 19.751, 6.2e-4, 106 |
| 2 | 49.348 | 50.811, 3.0e-2, 38 | 2.7e-2, 23 | 49.412, 1.3e-3, 122 |
| 3 | 49.348 | 49.754, 8.2e-3, 47 | 1.1e-2, 18 | 49.403, 1.1e-3, 125 |
| 4 | 78.957 | 79.256, 3.8e-3, 64 | 1.9e-3, 56 | 78.974, 2.2e-4, 137 |
| 5 | 98.696 | 98.856, 1.6e-3, 72 | 7.3e-3, 34 | 98.747, 5.2e-4, 157 |
| 6 | 98.696 | 98.771, 7.6e-4, 65 | 3.6e-3, 85 | 98.738, 4.3e-4, 169 |
| total | | 317 s | 241 s | 817 s |

Early stopping with tol 1e-5 on the stochastic loss stops at ~600-1300 epochs and leaves up to 3% error in the 2nd eigenvalue (the deflated eigenfunctions also
overlap 1-6% with predecessors); without early stopping errors are 2e-4..1.3e-3. Our scan:

| method | nodes | wall (train) | + refinement | result |
|---|---|---|---|---|
| scan [15,105] step 5, mu=450 (seed 0) | 19 | 2077 s | 99 s | 4 minima (2pi^2, 5pi^2, 8pi^2, 10pi^2; multiplicity not resolved), rel. err 1.2e-5, 1.3e-5, 1.2e-5, 2.4e-5, certificate r 0.16-0.94 |
| paper protocol scan [0,105], 80 nodes | 80 | 6534 s | 413 s | same 4, rel err 1e-6..3.5e-5 |
| narrow window [95,102], 5 nodes, cold start, mu=5000 | 5 | 950 s | 24 s | 10pi^2: E=98.69395, rel err 2.1e-5, r=1.06 (KT bound 0.056) |
| same, mu=450 | 5 | 348 s | - | collapse u=0 at every node (cold start needs mu >~ E^2/2) |

**Honest conclusion**: in 2D on the unit square with these settings the Rayleigh-quotient deflation approach is *much cheaper* for the lowest K=6 eigenvalues
(241-317 s with early stopping, 817 s without) than our scan (2000-6500 s) and is also resolving multiplicities (5pi^2, 10pi^2 appear twice). To reach 10pi^2 (5th eigenvalue) Rayleigh needs
253 s (ES) / 647 s (no ES) cumulative; our narrow window scan 950 s (with large mu for the cold start). The scan's advantages are not cost for the lowest modes: it is embarrassingly parallel across
windows/nodes (not exploited here), gives an a-posteriori certificate (r, Weinstein) per node without needing lower modes, and its cost does not grow with the index k - but the Rayleigh cost also did not grow
strongly for k<=6 here (error accumulation through deflation does: 2nd eigenvalue error 3% with ES). A fair statement for the paper: cost of the scan ~ (#nodes) x (cost of one training), #nodes ~ window/h.

## Caveats
* mu-sweep used the reduced network (validated at mu=1000 only). Single seed for Table-1 sweep and cost numbers; 2 seeds for mu; 3 for seeds.
* queue3 (narrow mu=450) was by mistake started concurrently with the 80-node sweep for ~6 min (the sweep's wall time is slightly affected, 4-core machine); a container restart interrupted the broad scan s0 (resumed).
* 3-point parabola was evaluated on coarse spacings only. Early-stopping makes wall times seed dependent (epochs per node 300-5000).
* Certificate r computed with 5e4 fresh points: noise ~1%.
