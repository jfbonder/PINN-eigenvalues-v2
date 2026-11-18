"""Chunked (batch-means) MC evaluation of a trained network: Rayleigh quotients with standard errors, r, mass.

The computation now lives in pinn_eigen.a_posteriori (same sample, same formulas, identical values);
this wrapper keeps the original interface and keys."""
import pinn_eigen as pe

_KEYS = ("rho_res", "rho_res_se", "rho_var", "rho_var_se", "r", "r_se", "mass", "mass_se", "res_mean")


def mc_eval(model, sampler, V_fn, E, n_chunks=40, chunk=10000, seed=777):
    ev = pe.a_posteriori(model, sampler, V_fn, E, p=2.0, n_chunks=n_chunks, chunk=chunk, seed=seed)
    return {k: ev[k] for k in _KEYS}
