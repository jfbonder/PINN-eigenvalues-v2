import json, sys, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from experiments.cross import cross_lib as C
RES = Path("experiments/results"); FIG = Path("experiments/cross/figures")
def hist_fig(E, variants, fn):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for v in variants:
        p = RES / f"cross_diag_cross_E{E}_{v}_s0.json"
        if not p.exists(): continue
        h = np.array(json.load(open(p))["hist"]); ax[0].semilogy(h[:, 0], label=v, lw=0.8); ax[1].semilogy(h[:, 1] + 1e-6, lw=0.8, label=v); ax[2].plot(1 - np.sqrt(h[:, 2]) * np.sign(1), lw=0.8)
    ax[0].set_title("total training loss per epoch"); ax[1].set_title("residual term mean(res^2)"); ax[2].set_title("~ mean(u^2) (1 - |mean u^2 - 1|)")
    for a in ax: a.set_xlabel("epoch")
    ax[0].legend(fontsize=7); fig.suptitle(f"cross, E={E}, mu=300, 5000 pts (cold start)"); fig.tight_layout(); fig.savefig(FIG / fn, dpi=120)
def resid_fig(E, v, fn, beps, enrich):
    B = partial(C.B_cross, eps=beps); m = C.make_model(B, corners=C.CORNERS_CROSS, enrich=enrich)
    m.load_state_dict(torch.load(RES / "cross_models" / f"cross_E{E}_{v}_s0.pt"))
    g = torch.linspace(-1, 1, 321); X, Y = torch.meshgrid(g, g, indexing="ij"); P = torch.stack([X.ravel(), Y.ravel()], 1)
    ins = C.in_cross(P); P = P[ins]
    u, rs, _ = m.u_res(P.clone(), E); u = u.detach().numpy().ravel(); rs = rs.detach().numpy().ravel()
    full = np.full(321 * 321, np.nan); fu = full.copy(); full[ins.numpy()] = np.abs(rs); fu[ins.numpy()] = u
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im = ax[0].imshow(fu.reshape(321, 321).T, origin="lower", extent=[-1, 1, -1, 1]); plt.colorbar(im, ax=ax[0]); ax[0].set_title(f"trained u ({v}), E={E}")
    im = ax[1].imshow(np.log10(full.reshape(321, 321).T + 1e-6), origin="lower", extent=[-1, 1, -1, 1]); plt.colorbar(im, ax=ax[1]); ax[1].set_title("log10 |residual| (Delta+E)u")
    cx = np.array(C.CORNERS_CROSS); ax[1].plot(cx[:, 0], cx[:, 1], "r.", ms=4)
    fig.tight_layout(); fig.savefig(FIG / fn, dpi=120)
    d = np.hypot(*(np.array(P)[:, None, :] - cx[None]).transpose(2, 0, 1)).min(1)
    near = d < 0.1
    print(v, "share of sum res^2 within 0.1 of corners:", float((rs[near] ** 2).sum() / (rs ** 2).sum()), "area share", float(near.mean()))
if __name__ == "__main__":
    hist_fig(26.1, ["base", "fixed", "decay", "n20000", "eps", "mu3000", "enr", "enr_eps", "enr_eps_fixed", "enr_eps_decay"], "diag_hist_E26.1.png")
    resid_fig(26.1, "enr_eps_fixed", "resid_enr_eps_fixed.png", 0.02, True)
    resid_fig(26.1, "enr_eps", "resid_enr_eps.png", 0.02, True)
