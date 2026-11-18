"""Job definitions for the p-Laplacian study.  Two training protocols:

  P0 ("paper"):    Adam, lr=1e-3 constant, <=2000 epochs, early stopping (patience 300, tol 1e-5)  [as published]
  P1 ("decay"):    Adam, lr 1e-3 -> 1e-5 cosine decay over a fixed number of epochs, no early stopping

Correct loss:  |O|*mean((Delta_p u + E|u|^{p-2}u)^2) + mu*(|O|*mean(|u|^p) - 1)^2, mu = 200.
Legacy loss (published results): mean((Delta_p u + E u)^2) + mu*(mean(u^2) - 1)^2, mu = 4000.
"""
import numpy as np

NOSTOP = 10 ** 9
P0 = dict(epochs=2000, patience=300, tol=1e-5)
P1 = dict(epochs=1500, patience=NOSTOP, lr_decay="cos")
BASE = dict(width=128, depth=3, N=5000, mu=200.0, seed=0)


def J(name, domain, p, E, proto, **kw):
    d = dict(BASE); d.update(proto); d.update(name=name, domain=domain, p=p, E=[float(e) for e in E]); d.update(kw)
    return d


G3 = [6.0, 6.5, 7.0]
KEEP = dict(keep_models=True)
LOC = dict(h=0.5, npts=3)
# higher eigenvalues (authors' request): extended windows, then a local 3-point scan at EVERY detected minimum,
# warm-started from the coarse-scan network of the nearest grid point.
HI_A = [
    J("disk22_coarse2_P1", "disk", 2.2, np.arange(24, 58.01, 1.0), P1, **KEEP),
    dict(J("disk22_loc", "disk", 2.2, [], P1), local_all_from=["disk22_coarse_P1", "disk22_coarse2_P1"], **LOC),
]
HI_B = [
    J("square22_coarse2_P1", "square2", 2.2, np.arange(19, 37.01, 1.0), P1, **KEEP),
    dict(J("square22_loc", "square2", 2.2, [], P1), local_all_from=["square22_coarse_P1", "square22_coarse2_P1"], **LOC),
]
A = [   # worker 0
    J("disk22_coarse_P1", "disk", 2.2, np.arange(3, 25.01, 1.0), P1),
    *HI_A,
    J("disk22_coarse_P0", "disk", 2.2, np.arange(3, 25.01, 1.0), P0),
    J("disk22_legacy_P0", "disk", 2.2, np.arange(5.5, 8.01, 0.25), P0, legacy=True, mu=4000.0),
    J("disk22_legacy_P1", "disk", 2.2, np.arange(6.0, 8.01, 0.5), P1, legacy=True, mu=4000.0),
    J("conv_P0_s0", "disk", 2.2, G3, P0, seed=0),
    J("conv_P0_s1", "disk", 2.2, G3, P0, seed=1),
    J("conv_P0_s2", "disk", 2.2, G3, P0, seed=2),
    J("conv_w64d2", "disk", 2.2, G3, P1, width=64, depth=2),
    J("conv_w32d2", "disk", 2.2, G3, P1, width=32, depth=2),
    J("conv_N2000", "disk", 2.2, G3, P1, N=2000),
    J("conv_ep500", "disk", 2.2, G3, P1, epochs=500),
    J("conv_mu50", "disk", 2.2, G3, P1, mu=50.0),
    J("conv_mu1000", "disk", 2.2, G3, P1, mu=1000.0),
    J("conv_N20000", "disk", 2.2, G3, P1, N=20000),
    J("conv_ep5000", "disk", 2.2, G3, P1, epochs=5000),
    J("disk_p1.5", "disk", 1.5, [3.0, 3.5, 4.0, 4.5, 5.0], P1),
    J("disk_p4_w256d4", "disk", 4.0, [14.0, 15.0, 16.0], P1, width=256, depth=4),
]
B = [   # worker 1
    J("square22_coarse_P1", "square2", 2.2, np.arange(3, 20.01, 1.0), P1, **KEEP),
    *HI_B,
    J("conv_base_s0", "disk", 2.2, [5.5, 6.0, 6.5, 7.0, 7.5], P1, seed=0),
    J("conv_base_s1", "disk", 2.2, G3, P1, seed=1),
    J("conv_base_s2", "disk", 2.2, G3, P1, seed=2),
    J("conv_h025", "disk", 2.2, [6.25, 6.5, 6.75], P1),
    J("conv_h0125", "disk", 2.2, [6.375, 6.5, 6.625], P1),
    J("ball25_coarse_P1", "ball3", 2.5, np.arange(8, 46.01, 2.0), P1),
    dict(J("ball25_E1_local", "ball3", 2.5, [], P1), local_from="ball25_coarse_P1", k=0, h=1.0, npts=3),
    dict(J("ball25_E2_local", "ball3", 2.5, [], P1, epochs0=3000), local_from="ball25_coarse_P1", k=1, h=1.0, npts=3),
    J("ball25_legacy_P1", "ball3", 2.5, [16.0, 17.0, 18.0, 19.0], P1, legacy=True, mu=4000.0),
    J("conv_w256d4", "disk", 2.2, G3, P1, width=256, depth=4),
    J("disk_p3", "disk", 3.0, [8.5, 9.25, 10.0, 10.75, 11.5], P1),
    J("disk_p4", "disk", 4.0, [13.0, 14.0, 15.0, 16.0, 17.0], P1),
    J("square22_coarse_P0", "square2", 2.2, np.arange(3, 20.01, 1.0), P0),
]
QUEUES = {0: A, 1: B}

# ---------------- Round 2 (referee check items 9, 13): more seeds, E2 convergence row, node dependence ----------------
GE2 = [18.0, 18.5, 19.0]
R2 = []
for s in (1, 2):
    R2 += [J(f"r2_diskE2_s{s}", "disk", 2.2, GE2, P1, seed=s, epochs0=3000),
           J(f"r2_sqE1_s{s}", "square2", 2.2, [5.0, 5.5, 6.0], P1, seed=s, epochs0=3000),
           J(f"r2_sqE2_s{s}", "square2", 2.2, [14.75, 15.25, 15.75], P1, seed=s, epochs0=3000)]
R2 += [J("r2_diskE2_w64d2", "disk", 2.2, GE2, P1, width=64, depth=2, epochs0=3000),
       J("r2_diskE2_P0", "disk", 2.2, GE2, P0),
       J("r2_diskE2_ep5000", "disk", 2.2, GE2, P1, epochs=5000)]
for s in (1, 2):
    R2 += [J(f"r2_disk_p3_s{s}", "disk", 3.0, [8.5, 9.25, 10.0, 10.75, 11.5], P1, seed=s),
           J(f"r2_disk_p4_s{s}", "disk", 4.0, [13.0, 14.0, 15.0, 16.0, 17.0], P1, seed=s)]
QUEUES[2] = R2
# legacy loss near the published second value 19.44 of the disk (true E2 = 18.393), for item 13
QUEUES[3] = [J("r2_disk22_legacyE2_P1", "disk", 2.2, [17.0, 18.0, 19.0, 20.0, 21.0], P1, legacy=True, mu=4000.0, epochs0=3000)]
