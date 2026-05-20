"""
myconet.freiman
===============
Local Freiman index for 2D mycorrhizal networks using k-nearest-neighbor
local sets. Uses hex-integer coordinates so that K_HEX = 19/7 exactly
(Lemma 4.1, Appendix A of the companion paper).

Key properties:
  - Exactly k nodes per local set (displacement-robust)
  - K_HEX = 19/7 ≈ 2.714 for the hexagonal lattice (k=7, Lemma 4.1)
  - sigma_r > K_HEX for any displacement from hexagonal order
  - sigma_r >> K_HEX for random or stress-induced irregular networks
"""
import numpy as np
from scipy.spatial import KDTree

# ── Theoretical constants (Appendix A) ──────────────────────────────────────
K_HEX  = 19 / 7        # ≈ 2.714  hex lattice local doubling constant
C1     = 24 / 7        # upper bound slope (Lemma 4.2)
c0     = 7  / 24       # Prop 4.4: W2 >= c0*eps*(sigma_r - K_HEX)
C_STAR = 256 * 49 / 576   # ≈ 21.78  Theorem 6.1 analytic constant


def _cart_to_hex(pts, eps):
    """
    Project points to hex-integer coordinates (i,j) via
        p ≈ i·e1 + j·e2,   e1 = ε(1,0),  e2 = ε(1/2, √3/2).
    Two points within ε/2 of the same hex node share the same (i,j).
    """
    e1 = np.array([eps, 0.0])
    e2 = np.array([eps/2.0, eps*np.sqrt(3)/2.0])
    M  = np.column_stack([e1, e2])
    return np.round(pts @ np.linalg.inv(M).T).astype(int)


def _sumset_ratio(pts, eps):
    """
    |A+A| / |A| for pts on the hex-integer lattice.
    For the canonical 7-point hex neighborhood, returns exactly 19/7.
    """
    n = len(pts)
    if n < 2:
        return float(K_HEX)
    ij  = _cart_to_hex(pts, eps)
    pos = ij - ij.min(axis=0)
    Ni, Nj = pos[:,0].max()+2, pos[:,1].max()+2
    A = np.zeros((Ni, Nj), dtype=np.float32)
    for ci, cj in pos:
        A[ci, cj] = 1.0
    cnt_A  = int(A.sum())
    if cnt_A < 2:
        return float(K_HEX)
    FA     = np.fft.rfft2(A, s=(2*Ni, 2*Nj))
    AA     = np.fft.irfft2(FA*FA)
    cnt_AA = int((AA > 0.5).sum())
    return float(cnt_AA) / float(cnt_A)


def local_freiman_index(nodes, eps, k=7):
    """
    Local Freiman index sigma_r(Gamma) using k-nearest-neighbor local sets.

    For the hexagonal lattice with k=7, recovers K_HEX = 19/7 exactly.
    Increases monotonically as the network deviates from hexagonal order.

    Parameters
    ----------
    nodes : (N, 2) array, node positions [cm]
    eps   : float, mean nearest-neighbor spacing [cm]  (defines hex basis)
    k     : int, local set size (default 7 = center + 6 hex neighbors)

    Returns
    -------
    sigma  : float, mean local Freiman index
    values : (N,) array, per-node ratios
    """
    nodes  = np.asarray(nodes, dtype=np.float64)
    tree   = KDTree(nodes)
    k_eff  = min(k, len(nodes))
    _, idx = tree.query(nodes, k=k_eff)
    values = np.array([_sumset_ratio(nodes[row], eps) for row in idx])
    return float(values.mean()), values


def freiman_excess(nodes, eps, k=7):
    """sigma_r(Gamma) - K_HEX, the Freiman excess above hexagonal baseline."""
    sigma, _ = local_freiman_index(nodes, eps, k)
    return sigma - K_HEX


def w2_lower_bound(sigma_r, eps):
    """Proposition 4.4 lower bound: W2 >= c0*eps*(sigma_r - K_HEX)."""
    return c0 * eps * max(0.0, sigma_r - K_HEX)


def dissipation_lower_bound(sigma_r, eps, D):
    """Theorem 6.1 lower bound: Psi >= C_STAR*D/eps^2*(sigma_r - K_HEX)^2."""
    ex = max(0.0, sigma_r - K_HEX)
    return C_STAR * D / eps**2 * ex**2
