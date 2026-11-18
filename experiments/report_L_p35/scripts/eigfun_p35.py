"""p=3.5 disk: PINN first eigenfunction (node E=12.5) vs radial shooting profile."""
import json, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from experiments.plap35 import core35 as C
torch.set_num_threads(1)
p, d = 3.5, 2; q = p / (p - 1); c = (1 / d) ** (1 / (p - 1)) / q; r0 = 1e-8
f = lambda r, y: [np.sign(y[1]) * abs(y[1] / r) ** (1 / (p - 1)), -r * np.sign(y[0]) * abs(y[0]) ** (p - 1)]
sol = solve_ivp(f, [r0, 12], [1 - c * r0 ** q, -(r0 ** 2 / 2)], rtol=1e-12, atol=1e-15, method="DOP853", events=lambda r, y: y[0], dense_output=True)
z1 = sol.t_events[0][0]; E1 = z1 ** p
prof = lambda r: sol.sol(np.clip(r * z1, r0, None))[0]
# polar quadrature on the disk
nr, nt = 400, 256; xg, wg = np.polynomial.legendre.leggauss(nr); R = 0.5 * (xg + 1); wr = 0.5 * wg * R
T = np.linspace(0, 2 * np.pi, nt, endpoint=False); RR, TT = np.meshgrid(R, T, indexing="ij"); W = (wr[:, None] * np.ones(nt)[None] * 2 * np.pi / nt).ravel()
X = np.stack([(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()], 1)
us = prof(RR.ravel()); us /= (W @ np.abs(us) ** p) ** (1 / p)
OUT = Path("experiments/report_L_p35/fig"); CK = Path("experiments/plap35/ckpt")
runs = [("base_s0_E12.500", None, "método base"), ("feat0_s0_E12.500", "fixed", r"entrada singular $|x|^{p/(p-1)}$")]
rows = []; fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
n = 241; g = np.linspace(-1, 1, n); GX, GY = np.meshgrid(g, g, indexing="ij"); ins = GX ** 2 + GY ** 2 < 1
Pg = np.stack([GX[ins], GY[ins]], 1)
for i, (ck, feat, lab) in enumerate(runs):
    m = C.make_model(dict(p=p, variant="base", feat=feat)); m.load_state_dict(torch.load(CK / f"{ck}.pt")); m.eval()
    with torch.no_grad():
        up = m(torch.tensor(X, dtype=torch.float32))[0].numpy().ravel().astype(float)
        ug = m(torch.tensor(Pg, dtype=torch.float32))[0].numpy().ravel().astype(float)
    s = np.sign(W @ (up * us)); nrm = (W @ np.abs(up) ** p) ** (1 / p); up = s * up / nrm; ug = s * ug / nrm
    dd = up - us
    e2 = np.sqrt(W @ dd ** 2 / (W @ us ** 2)); ep = (W @ np.abs(dd) ** p / (W @ np.abs(us) ** p)) ** (1 / p); emax = np.abs(dd).max() / us.max()
    rows.append(dict(ckpt=ck, relL2=float(e2), relLp=float(ep), relmax=float(emax), u0_pinn=float(up.reshape(nr, nt)[0].mean()), u0_shoot=float(us[0])))
    print(rows[-1], flush=True)
    # radial profiles along 8 rays
    a = ax[i, 0]; rr = np.linspace(0, 1, 300)
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        with torch.no_grad():
            v = m(torch.tensor(np.stack([rr * np.cos(th), rr * np.sin(th)], 1), dtype=torch.float32))[0].numpy().ravel()
        a.plot(rr, s * v / nrm, lw=0.7, color="tab:blue", alpha=0.6)
    sr = prof(rr); sr /= (W @ np.abs(prof(RR.ravel())) ** p) ** (1 / p)
    a.plot(rr, sr, "k--", lw=1.5, label="shooting"); a.plot([], [], color="tab:blue", label="PINN (8 rayos)"); a.legend(); a.set_xlabel("r"); a.set_title(f"{lab}: perfiles radiales")
    a = ax[i, 1]; a.plot(rr, (np.interp(rr, R, up.reshape(nr, nt).mean(1)) - sr), label="media angular PINN − shooting"); a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("r"); a.set_title(f"diferencia radial (L2 rel. {100*e2:.2f}%)"); a.legend(fontsize=8)
    Z = np.full((n, n), np.nan); Z[ins] = ug - prof(np.hypot(Pg[:, 0], Pg[:, 1])) / (W @ np.abs(prof(RR.ravel())) ** p) ** (1 / p)
    vm = np.nanmax(np.abs(Z)); im = ax[i, 2].imshow(Z.T, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-vm, vmax=vm)
    ax[i, 2].set_title("PINN − shooting"); plt.colorbar(im, ax=ax[i, 2], fraction=0.046)
fig.suptitle(f"p = 3.5, disco unidad: autofunción PINN (nodo E = 12.5) vs shooting (E1 = {E1:.5f}), normalizadas ||u||_p = 1")
fig.tight_layout(); fig.savefig(OUT / "p35_eigfun.png", dpi=120)
json.dump(rows, open("experiments/results/plap35_eigfun.json", "w"), indent=1)
