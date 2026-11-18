"""Cross domain  Omega=[-1,1]x[-a,a] U [-a,a]x[-1,1] (a=1/4) and L-shape [-1,1]^2 \\ [0,1]^2: samplers, boundary factors, training loop, evaluation."""
import time, math
import numpy as np, torch
import pinn_eigen as pe

A = 0.25
AREA_CROSS = 4 * A * 1 - 4 * A * A      # 2*(2*2a) - (2a)^2 = 8a - 4a^2 = 1.75 for a=1/4
AREA_CROSS = 2 * (2 * 2 * A) / 2 * 1 + 0  # placeholder overwritten below
AREA_CROSS = 8 * A - 4 * A * A
AREA_L = 3.0
CORNERS_CROSS = [(sx * A, sy * A) for sx in (-1, 1) for sy in (-1, 1)]
CORNERS_L = [(0.0, 0.0)]

# ---------------------------------------------------------------- boundary factors
def B_cross(x, eps=0.0, scale=True):
    """R-disjunction of the two rectangle functions: phi1=(1-x^2)(a^2-y^2), phi2=(a^2-x^2)(1-y^2);
    B = phi1+phi2+sqrt(phi1^2+phi2^2+eps^2).  eps>0 smooths the kink at the re-entrant corners."""
    X, Y = x[:, 0:1], x[:, 1:2]
    p1 = (1 - X ** 2) * (A ** 2 - Y ** 2); p2 = (A ** 2 - X ** 2) * (1 - Y ** 2)
    big = torch.maximum(p1, p2); small = torch.minimum(p1, p2)
    s = torch.sqrt(p1 ** 2 + p2 ** 2 + eps ** 2)
    sm = torch.where(small < 0, (big ** 2 + eps ** 2) / (s - small), small + s)    # small+s, cancellation-free
    B = big + sm
    return B / 0.25 if scale else B

def B_L(x, eps=0.0):
    """L-shape [-1,1]^2 minus [0,1]^2: phi_sq=(1-x^2)(1-y^2) (square), phi_q=-(x*y)?? use R-conjunction of the square with
    the complement of the quadrant: psi = -max(x,0)... simple smooth choice: complement function g=(1-x)*(1... ) not smooth ->
    use R-conjunction  phi ^ psi = phi+psi-sqrt(phi^2+psi^2) with psi = x*? We use psi = -(x*y) smoothed (positive when xy<0)."""
    X, Y = x[:, 0:1], x[:, 1:2]
    phi = (1 - X ** 2) * (1 - Y ** 2)
    # complement of quadrant {x>0,y>0}: positive in the L (x<0 or y<0): R-disjunction of (-x) and (-y)
    s = -X - Y + torch.sqrt(X ** 2 + Y ** 2 + eps ** 2)       # >0 iff x<0 or y<0, =0 on the quadrant boundary rays
    return phi + s - torch.sqrt(phi ** 2 + s ** 2 + eps ** 2)

# ---------------------------------------------------------------- samplers
def sample_cross(n):
    """uniform in the cross: arm 1 = [-1,1]x[-a,a] (area 8a/... =1), arm 2 pieces |y|>a (area 0.75)."""
    a1 = 4 * A * 1.0 / 1.0 * 0 + 2 * 2 * A          # area of horizontal arm = 2 * 2a = 1.0
    a2 = 2 * (1 - A) * 2 * A                          # vertical pieces |y| in (a,1), |x|<a
    n1 = int(torch.distributions.Binomial(n, torch.tensor(a1 / (a1 + a2))).sample().item())
    x1 = torch.cat([2 * torch.rand(n1, 1) - 1, A * (2 * torch.rand(n1, 1) - 1)], 1)
    n2 = n - n1
    y = A + (1 - A) * torch.rand(n2, 1); y = y * (2 * (torch.rand(n2, 1) > 0.5).float() - 1)
    x2 = torch.cat([A * (2 * torch.rand(n2, 1) - 1), y], 1)
    return torch.cat([x1, x2], 0)

def sample_L(n):
    x = 2 * torch.rand(n * 2, 2) - 1
    m = ~((x[:, 0] > 0) & (x[:, 1] > 0))
    return x[m][:n]

def corner_sampler(base, inside, corners, R=0.15, frac=0.3):
    """Mixture: (1-frac) uniform + frac uniform in discs of radius R at the corners (restricted to the domain).
    Returns (points, weights) with weights = uniform density / mixture density (unbiased uniform means)."""
    def sampler(n):
        nc = int(round(frac * n)); xu = base(n - nc)
        k = torch.randint(0, len(corners), (nc * 3,))
        c = torch.tensor(corners)[k]
        r = R * torch.sqrt(torch.rand(nc * 3, 1)); th = 2 * math.pi * torch.rand(nc * 3, 1)
        xc = c + torch.cat([r * torch.cos(th), r * torch.sin(th)], 1)
        xc = xc[inside(xc)][:nc]
        while len(xc) < nc:
            xc = torch.cat([xc, corner_sampler.__wrapped_extra(inside, corners, R, nc - len(xc))], 0)
        x = torch.cat([xu, xc], 0)
        return x
    return sampler

def _extra(inside, corners, R, m):
    k = torch.randint(0, len(corners), (m * 4,)); c = torch.tensor(corners)[k]
    r = R * torch.sqrt(torch.rand(m * 4, 1)); th = 2 * math.pi * torch.rand(m * 4, 1)
    xc = c + torch.cat([r * torch.cos(th), r * torch.sin(th)], 1)
    return xc[inside(xc)][:m]
corner_sampler.__wrapped_extra = _extra

def in_cross(x):
    X, Y = x[:, 0].abs(), x[:, 1].abs()
    return ((X <= 1) & (Y <= A)) | ((X <= A) & (Y <= 1))
def in_L(x):
    return (x[:, 0].abs() <= 1) & (x[:, 1].abs() <= 1) & ~((x[:, 0] > 0) & (x[:, 1] > 0))

def mixture_weights(x, domain, corners, R, frac, area):
    """density ratio uniform/mixture at points x: uniform density 1/area; mixture = (1-frac)/area + frac * (disc density)
    where disc density(x) = sum_c 1[|x-c|<R] / (n_c * area_of(disc ∩ domain at c)) (approx: area of disc∩domain by MC, per corner)."""
    raise NotImplementedError

# ---------------------------------------------------------------- singular enrichment (analytic Laplacian)
import torch.nn as nn

class Enriched(nn.Module):
    """u = B(x) NN(x) + sum_c c_c chi(r_c) r_c^{2/3} sin(2 theta_c/3)   (c_c learnable, init 0).
    Re-entrant corner c of angle 3pi/2 at (sx*a, sy*a); theta_c is measured clockwise from the boundary ray along +x'
    in coordinates reflected so that the excluded quadrant is {dx'>0,dy'>0}.  The Laplacian of the singular part is analytic:
    Delta(chi s) = s (chi'' + 7/3 chi'/r), bounded (no autograd cancellation of r^{-4/3} terms)."""
    def __init__(self, B, corners, R=0.2, hidden=128, n_hidden=3, enrich=True, sym=False):
        super().__init__()
        self.core = pe.EigenFunctionNN(dim=2, boundary_fn=B, hidden_size=hidden, n_hidden=n_hidden)
        self.corners = corners; self.R = R; self.enrich = enrich
        self.c = nn.Parameter(torch.zeros(len(corners)))
    def sing(self, x):
        """returns (value, laplacian) of the singular part (no autograd needed)."""
        val = torch.zeros(x.shape[0], 1); lap = torch.zeros(x.shape[0], 1)
        for k, (cx, cy) in enumerate(self.corners):
            sx, sy = (1.0 if cx > 0 else -1.0), (1.0 if cy > 0 else -1.0)
            dx = sx * (x[:, 0:1] - cx); dy = sy * (x[:, 1:2] - cy)
            r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-14)
            th = torch.remainder(-torch.atan2(dy, dx), 2 * math.pi)          # clockwise from +x' ray, in [0,2pi)
            s = r ** (2.0 / 3) * torch.sin(2 * th / 3)
            rho2 = (r / self.R) ** 2; on = (r < self.R).float()
            chi = on * (1 - rho2) ** 2
            chi1 = on * (-4 * r / self.R ** 2 * (1 - rho2))
            chi2 = on * (-4 / self.R ** 2 * (1 - rho2) + 8 * r ** 2 / self.R ** 4)
            val = val + self.c[k] * chi * s
            lap = lap + self.c[k] * s * (chi2 + 7.0 / 3 * chi1 / r)
        return val, lap
    def u_res(self, x, E):
        x = x.requires_grad_(True)
        u = self.core(x); lap = pe._laplacian(u, x)
        if self.enrich:
            v, l = self.sing(x); u = u + v; lap = lap + l
        return u, lap + E * u, lap
    def forward(self, x):
        u = self.core(x)
        if self.enrich: u = u + self.sing(x)[0]
        return u

def make_enriched(B, corners, R=0.2, enrich=True):
    return Enriched(B, corners, R=R, enrich=enrich)

# ---------------------------------------------------------------- model / training
def make_model(B, hidden=128, n_hidden=3, corners=None, enrich=False, R=0.2):
    return Enriched(B, corners or [], R=R, hidden=hidden, n_hidden=n_hidden, enrich=enrich and bool(corners))

def train(model, E, sampler, mu, n_samples=5000, n_epochs=2000, lr=1e-3, patience=300, tol=1e-5, lr_decay=None,
          fixed_sample=False, record=True, seed=None):
    """Adam training at fixed E (loss = mean(res^2)+mu(mean(u^2)-1)^2, i.e. paper normalisation vol=1).
    lr_decay: None or gamma per epoch (ExponentialLR). fixed_sample: one batch for all epochs. Returns dict with per-epoch history."""
    if seed is not None: torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay) if lr_decay else None
    best, bad, hist = float("inf"), 0, []
    x0 = sampler(n_samples) if fixed_sample else None
    t0 = time.perf_counter()
    for ep in range(n_epochs):
        x = (x0 if fixed_sample else sampler(n_samples)).clone()
        opt.zero_grad()
        u, rs, _ = model.u_res(x, E)
        res = (rs ** 2).mean(); norm = (u ** 2).mean() - 1.0
        loss = res + mu * norm ** 2; res, norm = res.item(), norm.item() ** 2
        loss.backward(); opt.step()
        if sch: sch.step()
        l = loss.item()
        if record: hist.append((l, res, norm))
        if l < best - tol: best, bad = l, 0
        else: bad += 1
        if bad > patience: break
    return dict(hist=hist, epochs=ep + 1, wall=time.perf_counter() - t0)

def evaluate(model, sampler, E, n_chunks=20, chunk=10000, seed=777):
    """MC evaluation on fresh points in chunks: closed-form Rayleigh quotient, r, mass, with chunk-wise standard errors."""
    st = torch.get_rng_state(); torch.manual_seed(seed)
    S = []
    for _ in range(n_chunks):
        x = sampler(chunk); u, res, lap = model.u_res(x, E); u, res, lap = u.detach(), res.detach(), lap.detach()
        S.append([(u ** 2).mean().item(), (u * lap).mean().item(), (res ** 2).mean().item()])
    torch.set_rng_state(st)
    S = np.array(S); u2, ulu, r2 = S.T; m = S.mean(0)
    se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a)))
    return dict(rho=float(-m[1] / m[0]), rho_se=se(-ulu / u2), r=float(np.sqrt(m[2] / m[0])), r_se=se(np.sqrt(r2 / u2)),
                mass=float(m[0]), res_mean=float(m[2]), chunk_r=np.sqrt(r2 / u2).tolist(), chunk_rho=(-ulu / u2).tolist())
