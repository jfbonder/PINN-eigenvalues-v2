"""PINN vs FEM eigenfunctions on the L. usage: python -m experiments.cross.plot_eigfun_L OUT.png NAME1:ckpt1:k1 NAME2:ckpt2:k2 ...  (k = eigenvalue index 1..3)
Writes figure and a JSON with relative L2 errors (both normalized to unit L2 on the grid, sign aligned)."""
import sys, json
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from skfem import Basis, ElementTriP2
from experiments.cross import fem_cross as F, pen as P
torch.set_num_threads(1)
out = Path(sys.argv[1]); items = [s.split(":") for s in sys.argv[2:]]
m_ = F.graded("L", 2, 5); v, b, I, vec = F.solve(m_, 6, sigma=0.5); print("FEM", v[:4])
N = 401; g = np.linspace(-1, 1, N); X, Y = np.meshgrid(g, g, indexing="xy"); ins = F.inside_L(X, Y) & (np.abs(X) < 1) & (np.abs(Y) < 1)
pts = np.vstack([X[ins], Y[ins]]); pr = b.probes(pts)
def fem_fun(k):
    full = np.zeros(b.N); full[I] = vec[:, k - 1]; return pr @ full
def grid(vals): Z = np.full(X.shape, np.nan); Z[ins] = vals; return Z
fig, ax = plt.subplots(len(items), 3, figsize=(11, 3.4 * len(items)), squeeze=False); res = {}
for i, (nm, ck, k) in enumerate(items):
    k = int(k); m = P.make("L"); m.load_state_dict(torch.load(ck)); 
    with torch.no_grad(): u = m.core(torch.tensor(pts.T, dtype=torch.float32)).squeeze().numpy()
    f = fem_fun(k); u = u / np.sqrt((u ** 2).mean()); f = f / np.sqrt((f ** 2).mean())
    if (u * f).mean() < 0: u = -u
    err = np.sqrt(((u - f) ** 2).mean()); res[nm] = dict(rel_L2=float(err), fem_eig=float(v[k - 1]), ckpt=ck); print(nm, k, f"relL2={err:.4f}")
    vm = np.abs(f).max()
    for j, (Z, t, kw) in enumerate([(grid(u), f"PINN {nm}", dict(cmap="RdBu_r", vmin=-vm, vmax=vm)), (grid(f), f"FEM E{k}={v[k-1]:.4f}", dict(cmap="RdBu_r", vmin=-vm, vmax=vm)), (grid(np.abs(u - f)), f"|diff|, rel L2={err:.3f}", dict(cmap="viridis"))]):
        im = ax[i, j].imshow(Z, origin="lower", extent=[-1, 1, -1, 1], **kw); ax[i, j].set_title(t, fontsize=9); plt.colorbar(im, ax=ax[i, j], fraction=0.046)
plt.tight_layout(); plt.savefig(out, dpi=110); json.dump(res, open(str(out).replace("figures", "results").replace(".png", ".json").replace("/L_", "/cross_graded_L_"), "w"), indent=1)
