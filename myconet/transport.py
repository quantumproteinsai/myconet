"""
myconet.transport
=================
Fokker–Planck PDE solver, Wasserstein-2 distance, and Fisher information
estimator for the Freiman–Villani analysis.

Implements the thermodynamic observables of Section 6 and Appendix B of:
  Mercier des Rochettes, B. (2026). "Geometric Efficiency Bounds for
  Mycorrhizal Networks: A Freiman–Villani Framework."
"""

import numpy as np
from scipy.ndimage import gaussian_filter
try:
    import ot   # Python Optimal Transport
    _HAS_POT = True
except ImportError:
    _HAS_POT = False


# ─── Fokker–Planck PDE ───────────────────────────────────────────────────────

def fokker_planck_step(rho, vx, vy, source, D, dt, hx, hy):
    """
    One explicit time step of the Fokker–Planck equation:
        d_t rho = D Δrho − div(rho v) + f

    Uses central differences for diffusion and upwind scheme for advection.
    Neumann (zero-flux) boundary conditions.

    Parameters
    ----------
    rho    : (Nx, Ny) array, current probability density
    vx, vy : (Nx, Ny) arrays, drift field
    source : (Nx, Ny) array, nutrient source term f(x,t)
    D      : float, diffusion coefficient [cm^2/h]
    dt     : float, time step [h]
    hx, hy : float, grid spacings [cm]

    Returns
    -------
    rho_new : (Nx, Ny) array, updated density
    """
    # Diffusion: D * Laplacian(rho) — central differences
    lap = (
        (np.roll(rho, -1, axis=0) - 2*rho + np.roll(rho, 1, axis=0)) / hx**2 +
        (np.roll(rho, -1, axis=1) - 2*rho + np.roll(rho, 1, axis=1)) / hy**2
    )

    # Advection: -div(rho * v) — upwind
    rho_vx = rho * vx
    rho_vy = rho * vy
    # x-direction upwind
    flux_x_plus  = np.where(vx >= 0, rho_vx, np.roll(rho_vx, -1, axis=0))
    flux_x_minus = np.where(vx >= 0, np.roll(rho_vx, 1, axis=0), rho_vx)
    div_x = (flux_x_plus - flux_x_minus) / (2 * hx)
    # y-direction upwind
    flux_y_plus  = np.where(vy >= 0, rho_vy, np.roll(rho_vy, -1, axis=1))
    flux_y_minus = np.where(vy >= 0, np.roll(rho_vy, 1, axis=1), rho_vy)
    div_y = (flux_y_plus - flux_y_minus) / (2 * hy)

    rho_new = rho + dt * (D * lap - div_x - div_y + source)

    # Positivity and normalization
    rho_new = np.maximum(rho_new, 0)
    total = rho_new.sum() * hx * hy
    if total > 1e-14:
        rho_new /= total
    return rho_new


def growth_signal_step(c, rho, alpha, beta, Dc, dt, hx, hy):
    """
    One explicit step of the growth signal PDE:
        d_t c = alpha*rho - beta*c + Dc*Δc

    Parameters
    ----------
    c      : (Nx, Ny) array, growth signal concentration
    rho    : (Nx, Ny) array, nutrient concentration (source for c)
    alpha  : float, coupling from rho to c
    beta   : float, decay rate of c
    Dc     : float, diffusion of c [cm^2/h]
    """
    lap_c = (
        (np.roll(c, -1, axis=0) - 2*c + np.roll(c, 1, axis=0)) / hx**2 +
        (np.roll(c, -1, axis=1) - 2*c + np.roll(c, 1, axis=1)) / hy**2
    )
    c_new = c + dt * (alpha * rho - beta * c + Dc * lap_c)
    return np.maximum(c_new, 0)


# ─── Wasserstein-2 distance ───────────────────────────────────────────────────

def wasserstein2(nodes, hex_nodes, eps, reg=0.05, max_pts=300):
    """
    Approximate W_2(mu_Gamma, mu_hex) / eps using Sinkhorn algorithm (POT).

    Parameters
    ----------
    nodes     : (N, 2) array, network node positions
    hex_nodes : (M, 2) array, hexagonal reference lattice positions
    eps       : float, mean spacing [cm]
    reg       : float, Sinkhorn regularization
    max_pts   : int, subsample size for tractability

    Returns
    -------
    float : W_2(mu_Gamma, mu_hex) / eps  (dimensionless)
    """
    if not _HAS_POT:
        # Fallback: mean nearest-neighbor distance to hex lattice
        from scipy.spatial import KDTree
        tree = KDTree(hex_nodes)
        dists, _ = tree.query(nodes, k=1)
        return float(np.sqrt((dists**2).mean())) / eps

    # Subsample for speed
    rng = np.random.default_rng(42)
    n = min(len(nodes), max_pts)
    m = min(len(hex_nodes), max_pts)
    idx_n = rng.choice(len(nodes), n, replace=False)
    idx_h = rng.choice(len(hex_nodes), m, replace=False)
    pts_n = nodes[idx_n]
    pts_h = hex_nodes[idx_h]

    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(pts_n, pts_h)  # squared Euclidean cost matrix

    try:
        W2_sq = ot.sinkhorn2(a, b, M / eps**2, reg=reg, numItermax=500)[0]
        return float(np.sqrt(max(W2_sq, 0)))
    except Exception:
        # Fallback if Sinkhorn fails
        from scipy.spatial import KDTree
        tree = KDTree(pts_h)
        dists, _ = tree.query(pts_n, k=1)
        return float(np.sqrt((dists**2).mean())) / eps


# ─── Fisher information (metabolic dissipation proxy) ────────────────────────

def fisher_information(nodes, hex_nodes, eps, grid_size=40):
    """
    Estimate the Fisher information I(mu_Gamma | mu_hex) on a 2D grid:

        I = ∫ |∇ log(rho_Gamma / rho_hex)|^2 d mu_Gamma

    Parameters
    ----------
    nodes     : (N, 2) array, network node positions
    hex_nodes : (M, 2) array, hexagonal reference lattice
    eps       : float, mean spacing [cm]
    grid_size : int, resolution of evaluation grid

    Returns
    -------
    float : Fisher information estimate [1/cm^2]
    """
    domain = nodes.max(axis=0).max()
    x_grid = np.linspace(0, domain, grid_size)
    y_grid = np.linspace(0, domain, grid_size)
    hg = x_grid[1] - x_grid[0]
    XX, YY = np.meshgrid(x_grid, y_grid, indexing='ij')
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    # KDE for mu_Gamma (bandwidth 1.5*eps)
    bw_g = 1.5 * eps
    log_rho_g = np.zeros(len(pts))
    for nd in nodes:
        d2 = ((pts - nd)**2).sum(axis=1)
        log_rho_g += np.exp(-d2 / (2 * bw_g**2))
    log_rho_g = np.log(np.maximum(log_rho_g, 1e-30)).reshape(grid_size, grid_size)

    # KDE for mu_hex (bandwidth eps/4, matching Appendix B)
    bw_h = eps / 4.0
    log_rho_h = np.zeros(len(pts))
    for nd in hex_nodes:
        d2 = ((pts - nd)**2).sum(axis=1)
        log_rho_h += np.exp(-d2 / (2 * bw_h**2))
    log_rho_h = np.log(np.maximum(log_rho_h, 1e-30)).reshape(grid_size, grid_size)

    # Log-ratio and its gradient
    log_ratio = log_rho_g - log_rho_h
    grad_x = np.gradient(log_ratio, hg, axis=0)
    grad_y = np.gradient(log_ratio, hg, axis=1)
    grad_sq = grad_x**2 + grad_y**2

    # Weight by mu_Gamma (normalized)
    rho_g = np.exp(log_rho_g)
    rho_g /= rho_g.sum() * hg**2
    fisher = float((grad_sq * rho_g).sum() * hg**2)
    return fisher


def relative_entropy(nodes, hex_nodes, eps, grid_size=40):
    """
    Estimate H(mu_Gamma | mu_hex) = ∫ log(rho_Gamma/rho_hex) d mu_Gamma.

    Used to verify the Talagrand T2 bound: H >= (lambda_LS/2) * W2^2.
    """
    domain = nodes.max(axis=0).max()
    x_grid = np.linspace(0, domain, grid_size)
    hg = x_grid[1] - x_grid[0]
    XX, YY = np.meshgrid(x_grid, x_grid, indexing='ij')
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    bw_g = 1.5 * eps
    bw_h = eps / 4.0

    rho_g = np.zeros(len(pts))
    for nd in nodes:
        rho_g += np.exp(-((pts - nd)**2).sum(axis=1) / (2*bw_g**2))
    rho_g = np.maximum(rho_g, 1e-30)
    rho_g /= rho_g.sum() * hg**2

    rho_h = np.zeros(len(pts))
    for nd in hex_nodes:
        rho_h += np.exp(-((pts - nd)**2).sum(axis=1) / (2*bw_h**2))
    rho_h = np.maximum(rho_h, 1e-30)
    rho_h /= rho_h.sum() * hg**2

    log_ratio = np.log(rho_g) - np.log(rho_h)
    H = float((log_ratio * rho_g).sum() * hg**2)
    return max(H, 0.0)
