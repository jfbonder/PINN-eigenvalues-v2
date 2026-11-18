## 8. Round 2 (referee H: items 4, 5, 10, 15)

### 8.1 Per-seed worst case and FEM uncertainty (item 4)
| example | window | FEM ± est.err (mult) | other FEM eigenvalues within 0.75 | rho per seed | rel.err per seed (%) | worst |
|---|---|---|---|---|---|---|
| double_well | 8.1-11.1 | 9.5764 ± 5e-04 (1) | - | 9.5783, 9.5782 | +0.019, +0.018 | 0.019 |
| double_well | 18.4-21.4 | 19.7522 ± 3e-03 (1) | 19.967(x2) | 19.7640, 19.7547 | +0.060, +0.013 | 0.060 |
| double_well | 31.4-34.4 | 32.8735 ± 2e-02 (2) | 32.786(x1), 33.034(x2) | 32.9552, 32.8571 | +0.248, -0.050 | 0.248 |
| double_well | 37.7-40.7 | 39.1924 ± 1e-02 (1) | - | 39.1944, 39.1938 | +0.005, +0.003 | 0.005 |
| four_well | 8.3-11.3 | 9.8542 ± 5e-04 (1) | - | 9.8593, 9.8646, 9.8581 | +0.052, +0.105, +0.040 | 0.105 |
| four_well | 18.7-21.7 | 20.1801 ± 3e-03 (2) | 20.189(x1) | 20.1505, 20.1878, 20.1870 | -0.147, +0.038, +0.034 | 0.147 |
| four_well | 31.7-34.7 | 33.2104 ± 1e-02 (2) | 33.214(x3) | 33.2141, 33.2077, 33.2061 | +0.011, -0.008, -0.013 | 0.013 |
| four_well | 38.0-41.0 | 39.4573 ± 1e-02 (1) | - | 39.4578, 39.4414, 39.4554 | +0.001, -0.040, -0.005 | 0.040 |
| harm_w10 | 12.8-15.8 | 14.2847 ± 3e-07 (1) | - | 14.2899, 14.2904 | +0.036, +0.040 | 0.040 |
| harm_w10 | 27.6-30.6 | 29.1349 ± 2e-06 (2) | - | 29.1357, 29.1353 | +0.003, +0.001 | 0.003 |
| harm_w10 | 44.2-47.2 | 45.0887 ± 5e-06 (2) | - | 45.0892, 45.0892 | +0.001, +0.001 | 0.001 |
| harm_w10 | 61.1-64.1 | 62.5793 ± 2e-05 (2) | - | 62.5792, 62.5800 | -0.000, +0.001 | 0.001 |

Worst single-run error of the Rayleigh quotient over all windows/seeds: 0.248%. (Seed means are within 0.1%, single runs within 0.25%; the 0.25% case is double well 32.9, a window containing 3 FEM eigenvalues with spread 0.25, i.e. inside the cluster.) The FEM uncertainty (Richardson estimate, conservative) is 5e-4..2e-2 in 3-D and exceeds some of the PINN errors; for clusters the reference is the cluster, not a single member.

### 8.2 Blind scan of the double-well loss curve vs FEM (item 4)


### 8.3 Harmonic w=10, window extended to 49.2 (item 5)
| E | recorded loss | normalised r^2 | rho (net) ± SE | r | mass |
|---|---|---|---|---|---|
| 45.2 | 0.097 | 0.097 | 45.0930 ± 0.0004 | 0.312 | 0.971 |
| 45.7 | 0.439 | 0.446 | 45.0952 ± 0.0005 | 0.668 | 1.013 |
| 46.2 | 0.072 | 0.058 | 46.2903 ± 0.0004 | 0.240 | 1.016 |
| 46.7 | 0.196 | 0.191 | 46.2905 ± 0.0002 | 0.437 | 1.007 |
| 47.2 | 0.880 | 0.890 | 46.2919 ± 0.0004 | 0.943 | 1.002 |
| 47.7 | 1.931 | 2.086 | 46.2894 ± 0.0006 | 1.444 | 0.986 |
| 48.2 | 3.638 | 3.701 | 46.2912 ± 0.0004 | 1.924 | 0.977 |
| 48.7 | 5.693 | 5.932 | 46.2891 ± 0.0004 | 2.435 | 0.961 |
| 49.2 | 8.298 | 8.540 | 46.2904 ± 0.0004 | 2.922 | 0.962 |

### 8.4 Tables 2-4 re-run with local offset windows (item 10)
**Table 4 (triangle, mu=15000)**: 3 offset nodes (exact-0.9, +0.1, +1.1) per eigenvalue, fresh network at the first node, warm start along the window; rho = closed-form frozen-network Rayleigh quotient (4e5 MC pts, standard error in brackets).

| exact | rho (min-loss node) ± SE | rel.err | parabola on r^2 | rel.err | r ± SE | mass | min r^2 | paper value (v4) | paper rel.err |
|---|---|---|---|---|---|---|---|---|---|
| 49.3480 | 49.3509 (0.0014) | +0.0058% | 49.947 | +1.213% | 0.945 ± 0.002 | 0.997 | 0.893 | 49.3699 | +0.0444% |
| 98.6960 | 98.7067 (0.0024) | +0.0108% | nan | +nan% | 1.811 ± 0.003 | 1.011 | 3.279 | 98.8922 | +0.1987% |
| 128.3049 | 128.4612 (0.0095) | +0.1219% | nan | +nan% | 5.676 ± 0.015 | 0.986 | 32.214 | 128.0485 | -0.1998% |
| 167.7833 | 167.7923 (0.0035) | +0.0054% | nan | +nan% | 2.513 ± 0.003 | 1.006 | 6.314 | 167.6872 | -0.0573% |

All nodes (E: Rayleigh quotient, r, epochs; a node with r >> 1 or rho far from E has NOT converged to the targeted eigenfunction - cold start at the first node of each window):

- window 49.3480: 48.45: rho=55.965, r=44.20, ep=874; 49.45: rho=49.351, r=0.94, ep=1506; 50.45: rho=49.358, r=1.83, ep=925
- window 98.6960: 97.80: rho=57.121, r=47.41, ep=2000; 98.80: rho=99.044, r=4.46, ep=2000; 99.80: rho=98.707, r=1.81, ep=1655
- window 128.3049: 127.40: rho=88.028, r=61.92, ep=2000; 128.40: rho=135.850, r=20.51, ep=2000; 129.40: rho=128.461, r=5.68, ep=2000
- window 167.7833: 166.88: rho=164.660, r=15.01, ep=2000; 167.88: rho=167.735, r=3.37, ep=2000; 168.88: rho=167.792, r=2.51, ep=1793

**Table 2 (3-D ball, mu=100)**: 3 offset nodes (exact-0.9, +0.1, +1.1) per eigenvalue, fresh network at the first node, warm start along the window; rho = closed-form frozen-network Rayleigh quotient (4e5 MC pts, standard error in brackets).

| exact | rho (min-loss node) ± SE | rel.err | parabola on r^2 | rel.err | r ± SE | mass | min r^2 | paper value (v4) | paper rel.err |
|---|---|---|---|---|---|---|---|---|---|

All nodes (E: Rayleigh quotient, r, epochs; a node with r >> 1 or rho far from E has NOT converged to the targeted eigenfunction - cold start at the first node of each window):

- window 9.8696: 8.97: rho=9.870, r=0.92, ep=1689

### 8.5 Monte Carlo standard errors (item 15)
| run (saved network) | rho_res ± SE | rho_var ± SE | r ± SE | mean u^2 ± SE |
|---|---|---|---|---|
| harm_w10_s0_E45.200 | 45.0889 ± 0.0002 | 45.199 ± 0.068 | 0.233 ± 0.001 | 0.994 ± 0.002 |
| harm_w10_s0_E14.280 | 14.2882 ± 0.0006 | 14.314 ± 0.028 | 0.392 ± 0.001 | 0.936 ± 0.002 |
| four_well_s0_E9.850 | 9.8580 ± 0.0007 | 9.860 ± 0.022 | 0.577 ± 0.001 | 1.012 ± 0.002 |
| four_well_s0_E39.470 | 39.4519 ± 0.0024 | 39.447 ± 0.144 | 1.507 ± 0.003 | 1.014 ± 0.004 |
| double_well_s1_E19.860 | 19.7551 ± 0.0005 | 19.830 ± 0.045 | 0.272 ± 0.000 | 1.026 ± 0.002 |
| double_well_s1_E39.200 | 39.1942 ± 0.0005 | 39.192 ± 0.141 | 0.390 ± 0.001 | 1.003 ± 0.004 |