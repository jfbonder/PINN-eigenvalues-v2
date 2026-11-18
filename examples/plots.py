"""Figures of the examples: loss curves and computed eigenfunctions (saved as PNG, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ell2(E, ref):
    return np.min([(E - e) ** 2 for e, _, _ in ref], axis=0)


def plot_loss(ex, result, ax_list=None):
    """Recorded residual R_mu(E) (last training epoch) and normalized loss on fresh points, per window."""
    wins = result["windows"]
    n = len(wins)
    if ax_list is None:
        ncol = min(n, 4)
        nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol if n > 1 else 8, 3.4 * nrow), squeeze=False)
        ax_list = axs.ravel()
    else:
        fig = ax_list[0].figure
    for w, (ax, win) in enumerate(zip(ax_list, wins)):
        E = np.array(win["E"])
        rec = win["records"]
        if ex.p == 2.0:
            ax.plot(E, [r["loss_res"] for r in rec], "-o", ms=3, label=r"recorded residual $\mathcal{R}_\mu(E)$")
            ax.plot(E, [r["lam_n"] for r in rec], "-s", ms=2.5, alpha=0.7, label=r"normalized loss $r^2$ (fresh)")
            if ex.reference:
                t = np.linspace(E[0], E[-1], 600)
                l2 = _ell2(t, ex.reference)
                ax.plot(t, l2, ":", color="gray", label=r"$\ell(E)^2=\min_k(E-E_k)^2$")
                ax.plot(t, np.where(l2 <= 2 * ex.mu, l2 * (1 - l2 / (2 * ex.mu)), np.nan), "k--", lw=0.8,
                        label=r"$\ell^2(1-\ell^2/2\mu)$")
            ax.set_ylim(0, 1.5 * max(max(r["loss_res"] for r in rec), 1e-3))
        else:
            ax.semilogy(E, [r["lam_n"] for r in rec], "-o", ms=3, label=r"$\bar\Lambda_p(E)$ (fresh, normalized)")
        for e, m, _ in ex.reference:
            if E[0] - 0.5 <= e <= E[-1] + 0.5:
                ax.axvline(e, color="r", ls=":", lw=0.7)
        for d in result["detections"]:
            if d["window"] == w:
                ax.plot([d["E_node"]], [rec[d["node"]][ex.detect_on if ex.mode == "scan" else "lam_n"]],
                        "v", color="C3", ms=8)
        ax.set_xlabel("E")
        if w == 0:
            ax.legend(fontsize=7)
    for ax in ax_list[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{ex.name}: {ex.domain.name}, {ex.potential}" + (f", p={ex.p}" if ex.p != 2 else "")
                 + f" (mu={ex.mu:g})", fontsize=10)
    fig.tight_layout()
    return fig


def eigenfunction_on_plane(model, domain, n=121):
    """u on the plane x3 = ... = 0 over domain.box^2, NaN outside the domain."""
    a, b = domain.box
    lin = torch.linspace(a, b, n)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    XY = torch.zeros(n * n, domain.dim)
    XY[:, 0], XY[:, 1] = X.reshape(-1), Y.reshape(-1)
    with torch.no_grad():
        inside = (domain.boundary(XY) > 0).squeeze(-1)
        U = model(XY).squeeze(-1)
    U[~inside] = float("nan")
    return X.numpy(), Y.numpy(), U.reshape(n, n).numpy()


def plot_eigenfunctions(ex, result, states, max_panels=8):
    from examples.core import make_model
    dets = result["detections"][:max_panels]
    if not dets:
        return None
    k = len(dets)
    ncol = min(k, 4)
    nrow = int(np.ceil(k / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, d, st in zip(axs.ravel(), dets, states):
        m = make_model(ex)
        m.load_state_dict(st)
        X, Y, U = eigenfunction_on_plane(m, ex.domain)
        cf = ax.contourf(X, Y, U, levels=40, cmap="viridis")
        fig.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_aspect("equal")
        ax.set_title(rf"$\hat E$ = {d['E_hat']:.4f}", fontsize=9)
    for ax in axs.ravel()[k:]:
        ax.set_visible(False)
    sl = "" if ex.domain.dim == 2 else " (slice x3=...=0)"
    fig.suptitle(f"{ex.name}: computed eigenfunctions{sl}", fontsize=10)
    fig.tight_layout()
    return fig


def save_all(ex, result, states, out_dir, seed):
    out_dir = Path(out_dir)
    paths = []
    fig = plot_loss(ex, result)
    p = out_dir / f"{ex.name}_s{seed}_loss.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))
    fig = plot_eigenfunctions(ex, result, states)
    if fig is not None:
        p = out_dir / f"{ex.name}_s{seed}_eigenfunctions.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))
    return paths
