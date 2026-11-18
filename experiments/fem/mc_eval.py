"""Chunked (batch-means) MC evaluation of a trained network: Rayleigh quotients with standard errors, r, mass."""
import numpy as np, torch
import pinn_eigen as pe

def mc_eval(model, sampler, V_fn, E, n_chunks=40, chunk=10000, seed=777):
    st = torch.get_rng_state(); torch.manual_seed(seed)
    S = []
    for _ in range(n_chunks):
        x = sampler(chunk).requires_grad_(True)
        u = model(x)
        (gu,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        lap = pe._laplacian(u, x); V = V_fn(x).detach()
        res = lap - V * u + E * u
        S.append([(u**2).mean().item(), (gu**2).sum(1).mean().item(), (V*u**2).mean().item(),
                  (u*(lap - V*u)).mean().item(), (res**2).mean().item()])
    torch.set_rng_state(st)
    S = np.array(S); m = S.mean(0); n = n_chunks
    u2, g2, vu2, ulu, r2 = S.T
    rho_res = -ulu / u2; rho_var = (g2 + vu2) / u2; rr = np.sqrt(r2 / u2)
    # SE via chunk-wise ratio spread (jackknife-free batch means)
    se = lambda a: float(np.std(a, ddof=1) / np.sqrt(n))
    return dict(rho_res=float(-m[3] / m[0]), rho_res_se=se(rho_res), rho_var=float((m[1] + m[2]) / m[0]), rho_var_se=se(rho_var),
                r=float(np.sqrt(m[4] / m[0])), r_se=se(rr), mass=float(m[0]), mass_se=se(u2), res_mean=float(m[4]))
