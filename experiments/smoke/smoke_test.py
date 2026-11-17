"""Smoke tests: unit square (min near 2*pi^2) + p-Laplacian disk single E.
Run:  python -m experiments.smoke.smoke_test [--threads 2]"""
from functools import partial
import numpy as np
from experiments import common as C
import pinn_eigen as pe


def main():
    args = C.parse_args("smoke_square_and_pdisk")
    C.setup(args)
    out = {}

    # 1. unit square, Laplacian
    E = np.arange(15.0, 25.1, 1.0)
    sampler = partial(pe.sample_square, x0=0.0, x1=1.0)
    mf = lambda: pe.EigenFunctionNN(dim=2, boundary_fn=pe.boundary_square, hidden_size=64, n_hidden=3)
    with C.Timer() as t:
        res, models, recs = pe.sweep(mf, E, sampler, pe.V_zero, mu=100.0, reset_every=50,
                                     n_epochs=600, n_samples=2000, seed=args.seed,
                                     return_records=True, vol=pe.VOL_SQUARE, patience=10**9)
    k = int(np.argmin(res))
    print(f"square: min residual at E={E[k]} (exact 2pi^2={2*np.pi**2:.3f}); total {t.elapsed:.1f}s; "
          f"{t.elapsed/len(E):.2f}s per E")
    eigs, _ = pe.refine(models, E, res, sampler, pe.V_zero, threshold=1e9, vol=pe.VOL_SQUARE)
    C.save_results("smoke_square", dict(domain="unit square", hidden=64, n_hidden=3, n_epochs=600,
                   n_samples=2000, mu=100.0, E=E, seed=args.seed), recs, eigs,
                   dict(total_s=t.elapsed, per_E_s=t.elapsed / len(E)))

    # 2. p-Laplacian unit disk, p=2.2, single E
    p = 2.2
    sampler = partial(pe.sample_disk, radius=1.0)
    mf = lambda: pe.EigenFunctionNN(dim=2, boundary_fn=pe.boundary_disk, hidden_size=64, n_hidden=3)
    with C.Timer() as t:
        res, models, recs = pe.sweep(mf, np.array([12.0]), sampler, pe.V_zero, mu=100.0, p=p,
                                     n_epochs=600, n_samples=2000, seed=args.seed,
                                     return_records=True, vol=pe.VOL_DISK, patience=10**9)
    print("p-disk:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in recs[0].items()})
    C.save_results("smoke_pdisk", dict(p=p, E=12.0, hidden=64, n_epochs=600, n_samples=2000, mu=100.0),
                   recs, [], dict(total_s=t.elapsed))


if __name__ == "__main__":
    main()
