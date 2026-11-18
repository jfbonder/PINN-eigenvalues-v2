"""Job lists for the p = 3.5 study (two queues, one process each)."""
import json, os
import numpy as np
G5 = [10.5, 11.5, 12.5, 13.5, 14.5]            # E1 window (E1 = 12.1546), h = 1, not centred on the reference
G3 = [11.5, 12.5, 13.5]
FINE = [float(x) for x in np.arange(11.0, 14.01, 0.25)]
B0 = dict(p=3.5, mu=200.0, epochs=1500, N=5000, seed=0)


def J(name, variant, E, **kw):
    d = dict(B0); d.update(name=name, variant=variant, E=E); d.update(kw); return d


def e2_grid():
    r = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "plap35_refs.json")))
    c = float(np.floor(r["halfdisk"]["extrap"]))
    return [c - 4, c - 2, c, c + 2, c + 4]


A = [J("base_s0", "base", G5, keep_models=True), J("base_s1", "base", G5, seed=1), J("base_s2", "base", G5, seed=2),
     J("base_P0_s0", "base", G5, proto="P0", epochs=2000), J("base_P0_s1", "base", G5, proto="P0", epochs=2000, seed=1),
     J("base_P0_s2", "base", G5, proto="P0", epochs=2000, seed=2),
     J("base_P0c_s0", "base", G5, proto="P0c"),
     J("base_fine_P1", "base", FINE),
     J("base_N20000", "base", G3, N=20000),
     J("base_ep5000", "base", G3, epochs=5000),
     J("base_E2", "base", "E2", epochs0=3000, keep_models=True)]
B = [J("mixed_s0", "mixed", G5, lam=100.0, keep_models=True), J("mixed_s1", "mixed", G5, lam=100.0, seed=1),
     J("mixed_s2", "mixed", G5, lam=100.0, seed=2),
     J("mixed_lam10", "mixed", G5, lam=10.0), J("mixed_lam1000", "mixed", G5, lam=1000.0),
     J("wdown_k0.3", "wdown", G5, kappa=0.3, keep_models=True), J("wdiv_k0.3", "wdiv", G5, kappa=0.3),
     J("base_imp", "base", G5, imp=0.5, imp_r=0.3),
     J("base_mu1000", "base", G5, mu=1000.0),
     J("base_fine_P0", "base", FINE, proto="P0", epochs=2000),
     J("mixed_fine_P1", "mixed", FINE, lam=100.0),
     J("mixed_P0_s0", "mixed", G5, lam=100.0, proto="P0", epochs=2000),
     J("mixed_E2", "mixed", "E2", lam=100.0, epochs0=3000, keep_models=True)]
C = [J("feat0_s0", "base", G5, feat="fixed", keep_models=True), J("featc_s0", "base", G5, feat="learn"),
     J("feat0_s1", "base", G5, feat="fixed", seed=1), J("feat0_s2", "base", G5, feat="fixed", seed=2),
     J("feat0_fine", "base", FINE, feat="fixed")]
D = [J("base_E2_premass", "base", "E2", epochs0=3000, premass=500, keep_models=True),
     J("mixed_lam10_s1", "mixed", G5, lam=10.0, seed=1), J("mixed_lam10_s2", "mixed", G5, lam=10.0, seed=2),
     J("mixed_E2_premass", "mixed", "E2", lam=100.0, epochs0=3000, premass=500)]
QUEUES = {"A": A, "B": B, "C": C, "D": D}
