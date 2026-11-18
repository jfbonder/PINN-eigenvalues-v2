"""PINN (weak BC, beta=300, scan _ck) vs FEM eigenfunctions on the L. usage: python -m experiments.report_L_p35.scripts.eigfun_L 9.4:1 15.4:2 19.4:3"""
import sys, json, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from experiments.cross import fem_cross as F, cross_lib as C, pen as P
torch.set_num_threads(1)
RES = Path("experiments/results"); OUT = Path("experiments/report_L_p35/fig")
pairs = [(float(a.split(":")[0]), int(a.split(":")[1])) for a in sys.argv[1:]]
m = F.graded("L", 2, 5); v, b, I, vec = F.solve(m, 6, sigma=0.5)
n = 241; g = np.linspace(-1, 1, n); X, Y = np.meshgrid(g, g, indexing="ij")
ins = F.inside_L(X.ravel(), Y.ravel()) & (np.abs(X.ravel()) < 1 - 1e-9) & (np.abs(Y.ravel()) < 1 - 1e-9)
pts = np.vstack([X.ravel()[ins], Y.ravel()[ins]]); Pr = b.probes(pts)
rows = []; fig, ax = plt.subplots(len(pairs), 3, figsize=(13, 4 * len(pairs)), squeeze=False)
for i, (E, k) in enumerate(pairs):
    full = np.zeros(b.N); full[I] = vec[:, k - 1]; uf = Pr @ full
    net = P.make("L"); net.load_state_dict(torch.load(RES / "cross_models" / f"cross_scan_L_b300_ck_s0_E{E:g}.pt"))
    x = torch.tensor(pts.T, dtype=torch.float32); up = net.u_res(x.clone(), E)[0].detach().numpy().ravel()
    uf /= np.sqrt((uf ** 2).mean()); up /= np.sqrt((up ** 2).mean())
    if (uf * up).sum() < 0: up = -up
    d = up - uf; err = float(np.sqrt((d ** 2).mean())); errmax = float(np.abs(d).max() / np.abs(uf).max())
    rc = np.hypot(pts[0], pts[1]) < 0.2; share = float((d[rc] ** 2).sum() / (d ** 2).sum())
    rows.append(dict(E_node=E, k=k, fem=float(v[k - 1]), relL2=err, relmax=errmax, corner_share=share))
    print(rows[-1], flush=True)
    for j, (f, t) in enumerate([(up, f"PINN, nodo E={E:g}"), (uf, f"FEM, E_{k}={v[k-1]:.4f}"), (d, f"PINN - FEM (L2 rel. {100*err:.1f}%)")]):
        Z = np.full(n * n, np.nan); Z[ins] = f; Z = Z.reshape(n, n)
        vm = np.nanmax(np.abs(Z)); cm = "RdBu_r"
        im = ax[i, j].imshow(Z.T, origin="lower", extent=[-1, 1, -1, 1], cmap=cm, vmin=-vm, vmax=vm)
        ax[i, j].set_title(t); plt.colorbar(im, ax=ax[i, j], fraction=0.046)
fig.tight_layout(); tag = "_".join(str(k) for _, k in pairs)
fig.savefig(OUT / f"L_eigfun_{tag}.png", dpi=120); json.dump(rows, open(RES / f"cross_eigfun_L_{tag}.json", "w"), indent=1)
