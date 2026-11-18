### References

| p | d | E1(B_1) shooting | 2nd radial |
|---|---|---|---|
| 1.5 | 2 | 4.017795 | 13.0737 |
| 2.0 | 2 | 5.783186 | 30.4713 |
| 2.2 | 2 | 6.531740 | 41.7886 |
| 2.5 | 2 | 7.710246 | 66.1125 |
| 3.0 | 2 | 9.831498 | 137.9339 |
| 4.0 | 2 | 14.681640 | 559.0275 |
| 1.5 | 3 | 6.371403 | 16.6013 |
| 2.0 | 3 | 9.869604 | 39.4784 |
| 2.2 | 3 | 11.469906 | 54.6063 |
| 2.5 | 3 | 14.111228 | 87.4923 |
| 3.0 | 3 | 19.201037 | 186.2603 |
| 4.0 | 3 | 32.214611 | 782.8237 |

| FEM (P1, Richardson) | p | finest mesh | extrapolated | order |
|---|---|---|---|---|
| disk | 2.0 | 5.78328 | 5.78319 | 2.00 |
| disk | 2.2 | 6.53187 | 6.53174 | 2.00 |
| halfdisk | 2.0 | 14.68272 | 14.68197 | 2.00 |
| halfdisk | 2.2 | 18.39393 | 18.39286 | 2.00 |
| square2 | 2.2 | 5.49519 | 5.49505 | 1.98 |
| rect1x2 | 2.2 | 15.28162 | 15.28076 | 1.98 |
| tri_diag | 2.2 | 15.14373 | 15.14289 | 2.00 |
| ball3_axi | 2.5 | 14.11159 | 14.11123 | 2.00 |
| halfball3_axi | 2.0 | 20.19206 | 20.19073 | 2.00 |
| halfball3_axi | 2.5 | 35.63469 | 35.63150 | 2.00 |
| disk | 1.5 | 4.01784 | 4.01780 | 2.00 |
| disk | 3.0 | 9.83182 | 9.83150 | 1.99 |
| disk | 4.0 | 14.68246 | 14.68163 | 1.99 |
| sector_pi/2 | 2.0 | 26.37724 | 26.37462 | 2.00 |
| sector_pi/2 | 2.2 | 34.81212 | 34.80823 | 2.00 |
| sector_pi/3 | 2.2 | 56.11997 | 56.10990 | 2.00 |
| sector_pi/4 | 2.2 | 82.30762 | 82.28724 | 2.03 |
| strip3 | 2.2 | 33.17340 | 33.17197 | 2.00 |
| unit_square | 2.2 | 25.25120 | 25.24871 | 2.02 |

### Corrected loss (protocol P1): detected eigenvalues

| case | reference (type) | coarse 3-pt vertex | err % | local 3-pt vertex | err % | E_ls | err % | Lam_min (fresh) | r=sqrt | core frac |
|---|---|---|---|---|---|---|---|---|---|---|
| disk p=2.2, window [3,58], minimum 1 | 6.5317 (radial shooting (1st eigenvalue)) | 6.550 | +0.28 | 6.554 | +0.34 | 6.5337 | +0.03 | 0.115 | 0.340 | 0.56 |
| disk p=2.2, window [3,58], minimum 2 | 18.3929 (FEM half disk (1 nodal diameter); Horak 18.395) | 18.433 | +0.22 | 18.414 | +0.11 | 18.4069 | +0.08 | 0.832 | 0.912 | 0.40 |
| disk p=2.2, window [3,58], minimum 3 | 34.8082 (FEM sector pi/2 (2 nodal diameters)) | 34.848 | +0.11 | 34.794 | -0.04 | 34.8043 | -0.01 | 0.783 | 0.885 | 0.56 |
| disk p=2.2, window [3,58], minimum 4 | 41.7886 (radial shooting, 1 nodal circle) | 42.697 | +2.17 | 42.454 | +1.59 | 41.9615 | +0.41 | 4.734 | 2.176 | 0.34 |
| disk p=2.2, window [3,58], minimum 5 | 56.1099 (FEM sector pi/3 (3 nodal diameters)) | 56.170 | +0.11 | 55.658 | -0.81 | 56.1166 | +0.01 | 1.656 | 1.287 | 0.55 |
| square p=2.2, window [3,37], minimum 1 | 5.4950 (FEM square (1st eigenvalue); Horak 5.4952) | 5.511 | +0.28 | 5.522 | +0.49 | 5.4902 | -0.09 | 0.217 | 0.466 | 0.31 |
| square p=2.2, window [3,37], minimum 2 | 15.1429 (FEM half square, diagonal nodal line; Horak 15.144) | 15.146 | +0.02 | 15.160 | +0.11 | 15.1440 | +0.01 | 0.197 | 0.444 | 0.11 |
| square p=2.2, window [3,37], minimum 3 | 25.2487 (FEM [0,1]^2 = 2^p E1 (2x2 cells)) | 25.232 | -0.07 | 25.319 | +0.28 | 25.2285 | -0.08 | 1.026 | 1.013 | 0.58 |
| square p=2.2, window [3,37], minimum 4 | 32.4595 (scaled FEM triangle (both diagonals nodal), 2^{p/2} x 15.1429) | 32.509 | +0.15 | 32.544 | +0.26 | 32.4615 | +0.01 | 0.935 | 0.967 | 0.03 |

### Paper protocol (P0) and legacy loss

| run | grid min | 3-pt vertex | E_ls | Lam at min | published | true | err % |
|---|---|---|---|---|---|---|---|
| disk p=2.2 correct loss, P0 (paper protocol) | 6.00 | 6.4949 | 6.5309 | 0.401 | -- | 6.5317 | -0.56 |
| disk p=2.2 correct loss, P0 (paper protocol) | 18.00 | 18.4019 | 18.3835 | 0.687 | -- | 18.3929 | +0.05 |
| disk p=2.2 LEGACY loss, P0 | 7.00 | 6.8753 | 6.9241 | 0.460 | 6.679 | 6.5317 | +5.26 |
| disk p=2.2 LEGACY loss, P1 | 7.00 | 6.9560 | 6.9342 | 0.365 | 6.679 | 6.5317 | +6.50 |

### Convergence study, disk p=2.2, E1 = 6.53174 (grid 6.0, 6.5, 7.0 unless stated)

| setting | Lam(6.0) | Lam(6.5) | Lam(7.0) | core frac | 3-pt vertex | err % | E_ls | err % | RQ | err % | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline P1: 3x128, N=5000, 1500 ep, mu=200, h=0.5 (seed 0) | 0.394 | 0.131 | 0.316 | 0.59 | 6.5433 | +0.18 | 6.5320 | +0.00 | 6.5864 | +0.84 | 18 |
| baseline, seed 1 | 0.416 | 0.130 | 0.329 | 0.54 | 6.5449 | +0.20 | 6.5329 | +0.02 | 6.4834 | -0.74 | 10 |
| baseline, seed 2 | 0.401 | 0.125 | 0.318 | 0.56 | 6.5443 | +0.19 | 6.5285 | -0.05 | 6.5938 | +0.95 | 10 |
| E-step h=0.25 | -- | 0.120 | -- | 0.56 | 6.5828 | +0.78 | 6.5338 | +0.03 | 6.5248 | -0.11 | 10 |
| E-step h=0.125 | -- | 0.116 | -- | 0.55 | 6.7727 | +3.69 | 6.5337 | +0.03 | 6.5247 | -0.11 | 10 |

### Other p (unit disk, E1, baseline P1)

| p | run | h | ref | 3-pt vertex | err % | E_ls | err % | RQ | err % | Lam_min | core frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.2 | conv_base_s0 | 0.5 | 6.5317 | 6.5433 | +0.18 | 6.5320 | +0.00 | 6.5864 | +0.84 | 0.131 | 0.59 |

Total training wall time (sum over jobs, 1 thread each): 8.6 h
