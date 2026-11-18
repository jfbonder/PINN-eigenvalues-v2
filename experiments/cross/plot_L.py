import json, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from experiments.cross import cross_lib as C, pen as P
RES = Path("experiments/results"); FIG = Path("experiments/cross/figures")
FEM = [9.6402, 15.1984, 19.7424, 29.5307, 31.9246, 41.4978, 44.9683]
def scan_fig(betas):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for b in betas:
        p = RES / f"cross_scan_L_b{b:g}_s0.json"
        if not p.exists(): continue
        R = json.load(open(p))["records"]; E = np.array([r["E"] for r in R])
        ax[0].plot(E, [r["final_loss"][1] for r in R], "o-", ms=3, label=f"beta={b:g}: recorded residual term")
        ax[0].plot(E, [r["r"] ** 2 for r in R], "s--", ms=3, label=f"beta={b:g}: r^2 (fresh MC)")
        ax[1].plot(E, [r["rho_res"] for r in R], "o-", ms=3, label=f"beta={b:g} rho_res"); ax[1].plot(E, [r["rho_var"] for r in R], "x:", label=f"beta={b:g} rho_var")
        ax[2].semilogy(E, [r["bdry_rel"] for r in R], "o-", ms=3, label=f"beta={b:g}")
    for a in ax:
        for f in FEM: a.axvline(f, color="r", lw=0.6, ls="--")
        a.set_xlabel("E"); a.set_xlim(6, 25)
    ax[0].set_title("loss curve (L-shape, penalty BC)"); ax[1].set_title("Rayleigh quotients of trained nets"); ax[1].plot([6, 25], [6, 25], "k:", lw=0.5); ax[2].set_title("boundary violation  mean_b u^2 / mean u^2")
    for a in ax: a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "L_penalty_scan.png", dpi=130)
def hist_fig():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for v, lab in [("base", "B(x), beta n/a, lr const"), ("enr", "B + enrichment"), ("B1", "B*r"), ("enrB1", "B*r + enrichment")]:
        p = RES / f"cross_diag_L_E9.64_{v}_s0.json"
        if p.exists():
            h = np.array(json.load(open(p))["hist"]); ax[0].semilogy(h[:, 1], lw=0.7, label=lab)
    for tag, lab in [("", "penalty beta=100, lr const"), ("_dec", "penalty beta=100, lr decay")]:
        p = RES / f"cross_pen_L_E9.64_b100{tag}_s0.json"
        if p.exists():
            h = np.array(json.load(open(p))["hist"]); ax[1].semilogy(h[:, 1], lw=0.7, label=lab + ": residual term")
    for b in (1000,):
        p = RES / f"cross_pen_L_E9.64_b{b}_s0.json"
        if p.exists(): h = np.array(json.load(open(p))["hist"]); ax[1].semilogy(h[:, 1], lw=0.7, label="penalty beta=1000, lr const: residual term")
    ax[0].set_title("L-shape, E=9.64, hard BC via B(x): per-epoch residual term"); ax[1].set_title("L-shape, E=9.64, weak BC: per-epoch residual term")
    for a in ax: a.legend(fontsize=7); a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(FIG / "L_epoch_histories.png", dpi=130)
def resid_map(name, E, fn):
    m = P.make("L"); m.load_state_dict(torch.load(RES / "cross_models" / name))
    g = torch.linspace(-1, 1, 321); X, Y = torch.meshgrid(g, g, indexing="ij"); Pt = torch.stack([X.ravel(), Y.ravel()], 1); ins = C.in_L(Pt); Pt = Pt[ins]
    u, rs, _ = m.u_res(Pt.clone(), E); u = u.detach().numpy().ravel(); rs = rs.detach().numpy().ravel()
    fu = np.full(321 * 321, np.nan); fr = fu.copy(); fu[ins.numpy()] = u; fr[ins.numpy()] = np.abs(rs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im = ax[0].imshow(fu.reshape(321, 321).T, origin="lower", extent=[-1, 1, -1, 1]); plt.colorbar(im, ax=ax[0]); ax[0].set_title("trained u")
    im = ax[1].imshow(np.log10(fr.reshape(321, 321).T + 1e-6), origin="lower", extent=[-1, 1, -1, 1]); plt.colorbar(im, ax=ax[1]); ax[1].set_title("log10 |(Delta+E)u|"); ax[1].plot([0], [0], "r.")
    fig.tight_layout(); fig.savefig(FIG / fn, dpi=120)
    d = np.hypot(Pt[:, 0].numpy(), Pt[:, 1].numpy()); near = d < 0.2
    print(name, "share of sum res^2 within 0.2 of the corner:", float((rs[near] ** 2).sum() / (rs ** 2).sum()), "area share", float(near.mean()))
if __name__ == "__main__":
    scan_fig([100, 300]); hist_fig()
    resid_map("pen_L_E9.64_b100_s0.pt", 9.64, "L_resid_pen_b100.png")
