"""
myconet.network
===============
Hyphal junction node set management, hexagonal reference lattice generation,
and network-geometry-induced drift field computation.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def hexagonal_lattice(eps, domain_size, rng=None):
    """
    Generate a hexagonal lattice with spacing eps inside [0, domain_size]^2.

    Parameters
    ----------
    eps         : float, lattice spacing [cm]
    domain_size : float, side length of square domain [cm]
    rng         : numpy Generator (optional)

    Returns
    -------
    nodes : (N, 2) array of lattice node positions
    """
    e1 = np.array([eps, 0.0])
    e2 = np.array([eps * 0.5, eps * np.sqrt(3) / 2])

    nodes = []
    n_max = int(domain_size / eps) + 3
    for i in range(-1, n_max + 1):
        for j in range(-1, n_max + 1):
            p = i * e1 + j * e2
            if 0 <= p[0] <= domain_size and 0 <= p[1] <= domain_size:
                nodes.append(p)
    return np.array(nodes)


def mean_spacing(nodes):
    """Estimate mean nearest-neighbor distance in a node set."""
    from scipy.spatial import KDTree
    tree = KDTree(nodes)
    dists, _ = tree.query(nodes, k=2)   # k=2: self + nearest neighbor
    return float(dists[:, 1].mean())


def network_drift(nodes, eps, grid_x, grid_y, D):
    """
    Compute the drift field v_Gamma(x) = D * grad log rho_nodes(x) on a 2D grid,
    where rho_nodes is the Gaussian KDE of the node positions.

    The Fokker-Planck equation is:
        d_t rho = D * Delta rho - div(rho * v_Gamma)

    with steady-state rho_inf proportional to mu_hex.

    Parameters
    ----------
    nodes  : (N, 2) array of node positions [cm]
    eps    : float, mean spacing [cm]
    grid_x : (Nx,) array, x-grid points
    grid_y : (Ny,) array, y-grid points
    D      : float, diffusion coefficient [cm^2/h]

    Returns
    -------
    vx, vy : (Nx, Ny) arrays, drift field components
    """
    Nx, Ny = len(grid_x), len(grid_y)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing='ij')
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Gaussian KDE at bandwidth eta = eps/4 (matches Appendix B)
    eta = eps / 4.0
    log_rho = np.zeros(len(pts))
    for nd in nodes:
        diff = pts - nd
        log_rho += np.exp(-0.5 * (diff**2).sum(axis=1) / eta**2)
    log_rho = np.log(np.maximum(log_rho, 1e-30)).reshape(Nx, Ny)

    h_x = grid_x[1] - grid_x[0]
    h_y = grid_y[1] - grid_y[0]
    d_log_dx = np.gradient(log_rho, h_x, axis=0)
    d_log_dy = np.gradient(log_rho, h_y, axis=1)

    vx = D * d_log_dx
    vy = D * d_log_dy
    return vx, vy


class HyphalNetwork:
    """
    A mycorrhizal hyphal junction node set with methods for growth,
    Freiman index computation, and morphological analysis.

    Parameters
    ----------
    nodes       : (N, 2) array, initial node positions [cm]
    domain_size : float, domain side length [cm]
    eps         : float, target mean hyphal spacing [cm]
    rng         : numpy Generator
    """

    def __init__(self, nodes, domain_size=10.0, eps=0.01, rng=None):
        self.nodes = np.array(nodes, dtype=np.float64)
        self.domain_size = domain_size
        self.eps = eps
        self.rng = rng if rng is not None else np.random.default_rng(0)

    @classmethod
    def random_seed(cls, N, domain_size=10.0, eps=0.01, rng=None):
        """Create a network from N randomly placed initial nodes."""
        rng = rng if rng is not None else np.random.default_rng(0)
        nodes = rng.uniform(0, domain_size, size=(N, 2))
        return cls(nodes, domain_size=domain_size, eps=eps, rng=rng)

    @classmethod
    def from_hexagonal(cls, domain_size=10.0, eps=0.01, noise=0.0, rng=None):
        """Create a network from a (possibly perturbed) hexagonal lattice."""
        rng = rng if rng is not None else np.random.default_rng(0)
        nodes = hexagonal_lattice(eps, domain_size)
        if noise > 0:
            nodes += rng.normal(0, noise, nodes.shape)
            nodes = np.clip(nodes, 0, domain_size)
        return cls(nodes, domain_size=domain_size, eps=eps, rng=rng)

    def branch(self, c_field, grid_x, grid_y, dt,
               alpha_b=0.5, sigma_noise=0.3, max_nodes=4000):
        """
        Stochastic branching update: add new nodes based on the growth signal c.

        Parameters
        ----------
        c_field    : (Nx, Ny) array, growth signal concentration
        grid_x/y   : 1D grid arrays for c_field
        dt         : float, time step [h]
        alpha_b    : float, branching rate coefficient
        sigma_noise: float, directional noise magnitude
        max_nodes  : int, maximum total nodes allowed
        """
        if len(self.nodes) >= max_nodes:
            return

        interp_c = RegularGridInterpolator(
            (grid_x, grid_y), c_field, method='linear', bounds_error=False, fill_value=0.0
        )
        grad_cx = np.gradient(c_field, grid_x[1] - grid_x[0], axis=0)
        grad_cy = np.gradient(c_field, grid_y[1] - grid_y[0], axis=1)
        interp_gcx = RegularGridInterpolator(
            (grid_x, grid_y), grad_cx, method='linear', bounds_error=False, fill_value=0.0
        )
        interp_gcy = RegularGridInterpolator(
            (grid_x, grid_y), grad_cy, method='linear', bounds_error=False, fill_value=0.0
        )

        new_nodes = []
        perm = self.rng.permutation(len(self.nodes))
        for i in perm:
            xi = self.nodes[i]
            c_local = float(interp_c(xi[np.newaxis]).ravel()[0])
            p_branch = alpha_b * max(c_local, 0) * dt
            if self.rng.random() < p_branch:
                gcx = float(interp_gcx(xi[np.newaxis]).ravel()[0])
                gcy = float(interp_gcy(xi[np.newaxis]).ravel()[0])
                direction = np.array([gcx, gcy])
                norm_d = np.linalg.norm(direction)
                if norm_d > 1e-10:
                    direction /= norm_d
                else:
                    direction = self.rng.normal(0, 1, 2)
                    direction /= np.linalg.norm(direction)
                direction += sigma_noise * self.rng.normal(0, 1, 2)
                direction /= np.linalg.norm(direction)
                new_pt = xi + self.eps * direction
                if 0 < new_pt[0] < self.domain_size and 0 < new_pt[1] < self.domain_size:
                    new_nodes.append(new_pt)
                    if len(self.nodes) + len(new_nodes) >= max_nodes:
                        break

        if new_nodes:
            self.nodes = np.vstack([self.nodes, new_nodes])

    def prune(self, c_field, grid_x, grid_y, dt, beta_p=0.1):
        """
        Remove nodes where growth signal is very low (hyphal retraction under stress).
        """
        interp_c = RegularGridInterpolator(
            (grid_x, grid_y), c_field, method='linear', bounds_error=False, fill_value=0.0
        )
        c_vals = interp_c(self.nodes)
        threshold = np.percentile(c_vals, 5)  # bottom 5%
        survive = (c_vals > threshold) | (self.rng.random(len(self.nodes)) > beta_p * dt)
        if survive.sum() >= 10:
            self.nodes = self.nodes[survive]

    @property
    def N(self):
        return len(self.nodes)
