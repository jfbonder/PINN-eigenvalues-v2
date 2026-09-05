"""
pinn_eigen.py
=============
Core module for "A PINNs approach for the computation of eigenvalues
in elliptic problems" (Fernández Bonder & Salort).

Provides:
  - EigenFunctionNN : neural network with automatic BC enforcement
  - loss_fn         : PINN loss (linear and p-Laplacian)
  - train_one       : train for a single E value
  - sweep           : scan a range of E values with warm initialization
  - refine          : detect and refine local minima of the loss curve
  - Samplers        : sample_disk, sample_square, sample_annulus,
                      sample_triangle, sample_ball_3d, sample_ball_4d
  - Boundary fns    : boundary_disk, boundary_square, boundary_triangle,
                      boundary_annulus, boundary_ball
  - Plotting        : plot_loss_curve, plot_eigenfunction_disk,
                      plot_eigenfunction_square, plot_eigenfunction_triangle
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize_scalar


# ═══════════════════════════════════════════════════════════════════════════════
#  Neural network
# ═══════════════════════════════════════════════════════════════════════════════

class EigenFunctionNN(nn.Module):
    """
    Fully-connected network that automatically satisfies homogeneous Dirichlet
    boundary conditions via multiplication by a user-supplied boundary factor.

    The network computes:
        u(x) = B(x) · NN(x)
    where B(x) vanishes on ∂Ω, so u = 0 on ∂Ω is guaranteed exactly.

    Parameters
    ----------
    dim : int
        Spatial dimension of the input (e.g. 2 for 2-D problems).
    boundary_fn : Callable[[Tensor], Tensor]
        Function B : ℝ^d → ℝ, shape [N, d] → [N, 1].
        Must satisfy B(x) = 0 for x ∈ ∂Ω and B(x) > 0 for x ∈ Ω.
    hidden_size : int
        Number of neurons per hidden layer (default: 128).
    n_hidden : int
        Number of hidden layers (default: 3).
    """

    def __init__(
        self,
        dim: int,
        boundary_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_size: int = 128,
        n_hidden: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)
        self.boundary_fn = boundary_fn
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_fn(x) * self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Differential operators
# ═══════════════════════════════════════════════════════════════════════════════

def _laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Laplacian Δu computed via autograd (sum of second-order diagonal terms)."""
    (grad_u,) = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )
    lap = sum(
        torch.autograd.grad(
            grad_u[:, i : i + 1],
            x,
            grad_outputs=torch.ones_like(grad_u[:, i : i + 1]),
            create_graph=True,
        )[0][:, i : i + 1]
        for i in range(x.shape[1])
    )
    return lap


def _p_laplacian(u: torch.Tensor, x: torch.Tensor, p: float) -> torch.Tensor:
    """p-Laplacian  div(|∇u|^{p-2} ∇u)  computed via autograd."""
    (grad_u,) = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )
    grad_norm = torch.norm(grad_u, dim=1, keepdim=True) + 1e-12  # avoid /0
    A = grad_norm ** (p - 2) * grad_u                            # [N, d]
    div = sum(
        torch.autograd.grad(
            A[:, i].sum(), x, create_graph=True
        )[0][:, i : i + 1]
        for i in range(x.shape[1])
    )
    return div


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss function
# ═══════════════════════════════════════════════════════════════════════════════

def loss_fn(
    model: nn.Module,
    x: torch.Tensor,
    E_val: float,
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    mu: float,
    p: float = 2.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    PINN loss for the eigenvalue problem
        -Δ_p u + V(x) u = E u    (p=2: standard Laplacian)
    or for the p-Laplacian (p≠2):
        -div(|∇u|^{p-2} ∇u) = E u    (V=0 assumed)

    The total loss is:
        L = ‖PDE residual‖²_{L²} + μ · (‖u‖²_{L²} - 1)²

    Parameters
    ----------
    model : EigenFunctionNN
    x : Tensor of shape [N, d]  (collocation points, no grad needed beforehand)
    E_val : float
    V_fn : Callable, potential function V : Tensor[N,d] → Tensor[N,1]
    mu : float, penalty weight for the normalization term
    p : float, exponent (p=2 → Laplacian; p≠2 → p-Laplacian)

    Returns
    -------
    loss : Tensor (total, used for .backward())
    residual : float (PDE term, detached)
    norm_err : float (normalization term, detached)
    """
    x = x.requires_grad_(True)
    u = model(x)
    V = V_fn(x).detach()

    if p == 2.0:
        Lu = _laplacian(u, x)
        pde_residual = Lu - V * u + E_val * u
    else:
        Lu = _p_laplacian(u, x, p)
        pde_residual = Lu - V * u + E_val * u   # linearised: |u|^{p-2}u ≈ u near p=2

    loss_res  = (pde_residual ** 2).mean()
    loss_norm = ((u ** 2).mean() - 1.0) ** 2
    total     = loss_res + mu * loss_norm

    return total, loss_res.item(), loss_norm.item()


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one(
    model: nn.Module,
    E_val: float,
    sampler: Callable[[int], torch.Tensor],
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 5000,
    n_epochs: int = 3000,
    lr: float = 1e-3,
    mu: float = 100.0,
    p: float = 2.0,
    patience: int = 300,
    tol: float = 1e-5,
) -> Tuple[float, float]:
    """
    Train *model* for a fixed eigenvalue parameter *E_val*.

    Returns
    -------
    residual : float  — final PDE residual
    norm_err : float  — final normalization error
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    no_improve = 0

    for _ in range(n_epochs):
        x = sampler(n_samples)
        optimizer.zero_grad()
        loss, res, norm = loss_fn(model, x, E_val, V_fn, mu, p)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - tol:
            best_loss  = loss.item()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > patience:
            break

    return res, norm


def sweep(
    model_factory: Callable[[], nn.Module],
    E_values: np.ndarray,
    sampler: Callable[[int], torch.Tensor],
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    mu: float = 100.0,
    p: float = 2.0,
    reset_every: int = 50,
    verbose: bool = True,
    **train_kwargs,
) -> Tuple[List[float], List[nn.Module]]:
    """
    Scan *E_values*, training the network at each point.

    Uses *warm initialization*: for most steps, the weights from the previous
    E value are reused as the starting point (speeds up convergence). Every
    *reset_every* steps the network is re-initialized from scratch (avoids
    getting stuck in a local minimum).

    Parameters
    ----------
    model_factory : Callable[[], nn.Module]
        Returns a freshly initialized EigenFunctionNN.
    E_values : np.ndarray
        Grid of E values to scan.
    sampler : Callable[[int], Tensor]
        Returns n collocation points inside Ω.
    V_fn : Callable
        Potential V(x).
    mu : float
        Normalization penalty weight. Tip: use mu ∝ E² for high eigenvalues.
    p : float
        Exponent (p=2 → Laplacian).
    reset_every : int
        Reset weights every this many steps.
    verbose : bool
    **train_kwargs
        Forwarded to train_one (e.g. n_epochs, lr, n_samples, patience).

    Returns
    -------
    residuals : List[float]
    trained_models : List[nn.Module]
    """
    residuals: list[float] = []
    trained_models: list[nn.Module] = []
    model = model_factory()

    for i, E in enumerate(E_values):
        # Warm initialization (or periodic reset)
        if i > 0 and i % reset_every != 0:
            new_model = model_factory()
            new_model.load_state_dict(model.state_dict())
            model = new_model
        else:
            model = model_factory()

        res, norm_err = train_one(model, E, sampler, V_fn, mu=mu, p=p, **train_kwargs)
        residuals.append(res)
        trained_models.append(model)

        if verbose:
            print(f"  E={E:8.3f}  residual={res:.4e}  norm_err={norm_err:.2e}")

    return residuals, trained_models


def refine(
    trained_models: List[nn.Module],
    E_values: np.ndarray,
    residuals: List[float],
    sampler: Callable[[int], torch.Tensor],
    V_fn: Callable[[torch.Tensor], torch.Tensor],
    threshold: float,
    n_samples: int = 10_000,
    p: float = 2.0,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Detect local minima of the loss curve below *threshold* and refine each
    one with a scalar optimization (scipy.minimize_scalar) using the already-
    trained model weights as a fixed function approximator.

    Parameters
    ----------
    trained_models, E_values, residuals : from sweep()
    threshold : float
        Only minima with loss < threshold are considered eigenvalue candidates.
    n_samples : int
        Collocation points used for the refinement objective.
    p : float

    Returns
    -------
    eigenvalues : List[(E_approx, residual)]
    indices : List[int]  — positions in E_values where minima were found
    """
    eigenvalues: list[tuple[float, float]] = []
    indices: list[int] = []

    for i in range(1, len(residuals) - 1):
        is_local_min = (
            residuals[i] < residuals[i - 1]
            and residuals[i] < residuals[i + 1]
            and residuals[i] < threshold
        )
        if not is_local_min:
            continue

        model = trained_models[i]

        def objective(E: float) -> float:
            x = sampler(n_samples)
            _, res, _ = loss_fn(model, x, E, V_fn, mu=0.0, p=p)
            return res

        result = minimize_scalar(
            objective,
            bounds=(E_values[i - 1], E_values[i + 1]),
            method="bounded",
        )
        eigenvalues.append((result.x, result.fun))
        indices.append(i)

    print("\n── Eigenvalues found ──────────────────────────────────────")
    for k, (E, res) in enumerate(eigenvalues, 1):
        print(f"  {k:2d}.  E ≈ {E:.6f}    residual ≈ {res:.2e}")

    return eigenvalues, indices


# ═══════════════════════════════════════════════════════════════════════════════
#  Samplers
# ═══════════════════════════════════════════════════════════════════════════════

def sample_disk(n: int, radius: float = 1.0) -> torch.Tensor:
    """Uniform samples inside the 2-D disk of given radius."""
    r = radius * torch.sqrt(torch.rand(n, 1))
    theta = 2 * np.pi * torch.rand(n, 1)
    return torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def sample_square(n: int, x0: float = 0.0, x1: float = 1.0) -> torch.Tensor:
    """Uniform samples inside [x0, x1]²."""
    return x0 + (x1 - x0) * torch.rand(n, 2)


def sample_annulus(n: int, r_inner: float, r_outer: float) -> torch.Tensor:
    """Uniform samples inside the 2-D annulus {r_inner ≤ ‖x‖ ≤ r_outer}."""
    r = torch.sqrt(r_inner ** 2 + (r_outer ** 2 - r_inner ** 2) * torch.rand(n, 1))
    theta = 2 * np.pi * torch.rand(n, 1)
    return torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def sample_triangle(n: int) -> torch.Tensor:
    """Uniform samples inside the triangle with vertices (0,0), (1,0), (0,1)."""
    u = torch.rand(n, 1)
    v = torch.rand(n, 1)
    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    return torch.cat([u, v], dim=1)


def sample_ball_3d(n: int, radius: float = 1.0) -> torch.Tensor:
    """Uniform samples inside the 3-D ball."""
    r = radius * torch.rand(n, 1) ** (1 / 3)
    phi   = torch.acos(1 - 2 * torch.rand(n, 1))   # polar angle
    theta = 2 * np.pi * torch.rand(n, 1)            # azimuthal angle
    return torch.cat([
        r * torch.sin(phi) * torch.cos(theta),
        r * torch.sin(phi) * torch.sin(theta),
        r * torch.cos(phi),
    ], dim=1)


def sample_ball_4d(n: int, radius: float = 1.0) -> torch.Tensor:
    """Uniform samples inside the 4-D ball."""
    r      = radius * torch.rand(n, 1) ** (1 / 4)
    theta1 = 2 * np.pi * torch.rand(n, 1)
    theta2 = torch.acos(2 * torch.rand(n, 1) - 1)
    theta3 = 2 * np.pi * torch.rand(n, 1)
    return torch.cat([
        r * torch.sin(theta2) * torch.sin(theta3) * torch.cos(theta1),
        r * torch.sin(theta2) * torch.sin(theta3) * torch.sin(theta1),
        r * torch.sin(theta2) * torch.cos(theta3),
        r * torch.cos(theta2),
    ], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Boundary functions
#  Each function takes x: Tensor[N, d] and returns Tensor[N, 1].
#  They satisfy B(x) = 0 on ∂Ω and B(x) > 0 inside Ω.
# ═══════════════════════════════════════════════════════════════════════════════

def boundary_disk(x: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """B(x) = 1 − ‖x‖²/r²  for the 2-D disk."""
    return 1.0 - (x[:, 0:1] ** 2 + x[:, 1:2] ** 2) / radius ** 2


def boundary_square(
    x: torch.Tensor, x0: float = 0.0, x1: float = 1.0
) -> torch.Tensor:
    """B(x) = (x−x0)(x1−x)(y−x0)(x1−y)  for the square [x0, x1]²."""
    return (
        (x[:, 0:1] - x0) * (x1 - x[:, 0:1])
        * (x[:, 1:2] - x0) * (x1 - x[:, 1:2])
    )


def boundary_square_2x2(x: torch.Tensor) -> torch.Tensor:
    """B(x) = x(2−x)y(2−y)  for [0,2]²."""
    return x[:, 0:1] * (2 - x[:, 0:1]) * x[:, 1:2] * (2 - x[:, 1:2])


def boundary_triangle(x: torch.Tensor) -> torch.Tensor:
    """B(x) = x · y · (1−x−y)  for triangle (0,0),(1,0),(0,1)."""
    return x[:, 0:1] * x[:, 1:2] * (1 - x[:, 0:1] - x[:, 1:2])


def boundary_annulus(
    x: torch.Tensor, r_inner: float, r_outer: float
) -> torch.Tensor:
    """B(x) = (‖x‖ − r_inner)(r_outer − ‖x‖)  for the annular region."""
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    return (r - r_inner) * (r_outer - r)


def boundary_ball(x: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """B(x) = 1 − ‖x‖²/r²  for balls in any dimension (3-D, 4-D, …)."""
    return 1.0 - x.pow(2).sum(dim=1, keepdim=True) / radius ** 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Potential functions
# ═══════════════════════════════════════════════════════════════════════════════

def V_zero(x: torch.Tensor) -> torch.Tensor:
    """Zero potential (Helmholtz / pure Dirichlet problem)."""
    return torch.zeros(x.shape[0], 1)


def V_harmonic(x: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """Harmonic potential V(x) = (ω²/2) ‖x‖²."""
    return 0.5 * omega ** 2 * x.pow(2).sum(dim=1, keepdim=True)


def V_double_well_gaussian(
    x: torch.Tensor,
    a: float = 1.0,
    V0: float = 1.0,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Double Gaussian well: V(x) = −V0·[exp(−‖x−a·e₁‖²/2σ²) + exp(−‖x+a·e₁‖²/2σ²)]
    where e₁ = (1,0,0,…).
    """
    shift = torch.zeros(1, x.shape[1])
    shift[0, 0] = a
    r1sq = ((x - shift) ** 2).sum(dim=1, keepdim=True)
    r2sq = ((x + shift) ** 2).sum(dim=1, keepdim=True)
    return -V0 * (torch.exp(-r1sq / (2 * sigma ** 2)) + torch.exp(-r2sq / (2 * sigma ** 2)))


def V_four_well_bump(
    x: torch.Tensor,
    a: float = 0.22,
    sigma: float = 0.2,
    V0: float = 1.0,
) -> torch.Tensor:
    """
    Compactly supported four-well potential (bump functions).
    Centers at (±a, ±a, 0) in 3-D.
    """
    centers = [(a, a, 0), (-a, a, 0), (a, -a, 0), (-a, -a, 0)]
    V = torch.zeros(x.shape[0], 1)
    for cx, cy, cz in centers:
        c = torch.tensor([[cx, cy, cz]], dtype=x.dtype)
        r = torch.sqrt(((x - c) ** 2).sum(dim=1, keepdim=True))
        rr = r / sigma
        mask = (rr < 1).squeeze()
        t = torch.zeros_like(r)
        t[mask] = torch.exp(-1.0 / (1.0 - rr[mask] ** 2))
        V -= V0 * t
    return V


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curve(
    E_values: np.ndarray,
    residuals: List[float],
    exact_eigenvalues: Optional[List[float]] = None,
    ylim: Optional[float] = None,
    title: str = "Loss curve",
) -> None:
    """
    Plot the discrete loss curve Λ̄(E) and optionally the theoretical
    upper bound min_k (E_k − E)².

    Parameters
    ----------
    E_values : np.ndarray
    residuals : list of float
    exact_eigenvalues : list of float, optional
        Known eigenvalues used to draw the theoretical upper bound.
    ylim : float, optional
        Upper y-axis limit.
    title : str
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(E_values, residuals, "-o", markersize=3, label=r"$\bar\Lambda(E)$")

    if exact_eigenvalues is not None:
        t = np.linspace(E_values[0], E_values[-1], 1000)
        ub = np.minimum.reduce([(t - Ek) ** 2 for Ek in exact_eigenvalues])
        ax.plot(t, ub, "--", label=r"$\min_k(E_k-E)^2$  (upper bound)", alpha=0.7)

    ax.set_xlabel(r"$E$")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_xlim(E_values[0], E_values[-1])
    ax.set_ylim(bottom=0, top=ylim)
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def _plot_2d(
    X: torch.Tensor,
    Y: torch.Tensor,
    U: np.ndarray,
    E_val: float,
) -> None:
    """Internal: side-by-side contour + 3-D surface for a 2-D eigenfunction."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121)
    cf = ax1.contourf(X.numpy(), Y.numpy(), U, levels=100, cmap="viridis")
    plt.colorbar(cf, ax=ax1)
    ax1.set_title(rf"Eigenfunction  $E = {E_val:.4f}$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X.numpy(), Y.numpy(), U, cmap="viridis", edgecolor="none")
    ax2.set_title(rf"$E = {E_val:.4f}$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u")
    ax2.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()


def plot_eigenfunction_disk(
    model: nn.Module,
    E_val: float,
    radius: float = 1.0,
    grid_size: int = 200,
) -> None:
    """Plot the eigenfunction on a 2-D disk."""
    lin = torch.linspace(-radius, radius, grid_size)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    inside = XY[:, 0] ** 2 + XY[:, 1] ** 2 <= radius ** 2

    with torch.no_grad():
        u = model(XY[inside]).squeeze().numpy()

    U = np.full(X.shape, np.nan)
    U_flat = U.flatten()
    U_flat[inside.nonzero(as_tuple=False).squeeze()] = u
    _plot_2d(X, Y, U_flat.reshape(X.shape), E_val)


def plot_eigenfunction_square(
    model: nn.Module,
    E_val: float,
    x0: float = 0.0,
    x1: float = 1.0,
    grid_size: int = 200,
) -> None:
    """Plot the eigenfunction on a 2-D square [x0, x1]²."""
    lin = torch.linspace(x0, x1, grid_size)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    with torch.no_grad():
        u = model(XY).squeeze().numpy()

    _plot_2d(X, Y, u.reshape(grid_size, grid_size), E_val)


def plot_eigenfunction_triangle(
    model: nn.Module,
    E_val: float,
    grid_size: int = 200,
) -> None:
    """Plot the eigenfunction on the triangle (0,0),(1,0),(0,1)."""
    lin = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    inside = (XY[:, 0] >= 0) & (XY[:, 1] >= 0) & (XY[:, 0] + XY[:, 1] <= 1)

    with torch.no_grad():
        u = model(XY[inside]).squeeze().numpy()

    U = np.full(X.shape, np.nan)
    U_flat = U.flatten()
    U_flat[inside.nonzero(as_tuple=False).squeeze()] = u
    _plot_2d(X, Y, U_flat.reshape(X.shape), E_val)
