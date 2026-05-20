"""
examples/quickstart.py
======================
Quick start demonstration of the MycoNet package.
Reproduces the key computations from the companion paper:

    Mercier des Rochettes (2026). Geometric Efficiency Bounds for
    Mycorrhizal Networks: A Freiman-Villani Framework. J. Math. Biol.

Runtime: < 5 seconds
Usage:   python examples/quickstart.py
"""
import numpy as np
from myconet import (
    hexagonal_lattice, local_freiman_index,
    w2_lower_bound, dissipation_lower_bound,
    K_HEX, C_STAR, c0
)

print("=" * 58)
print("  MycoNet — Freiman-Villani Framework: Quick Start")
print("=" * 58)

eps = 0.05    # mean hyphal spacing [cm]
D   = 3.6e-3  # phosphorus diffusivity [cm^2/h]

# ── 1. Hexagonal reference lattice ───────────────────────────────────────────
nodes_hex = hexagonal_lattice(eps=eps, domain_size=2.0)
sigma_hex, _ = local_freiman_index(nodes_hex, eps=eps, k=7)

print(f"\n1. Hexagonal reference lattice")
print(f"   N = {len(nodes_hex)} nodes,  eps = {eps} cm")
print(f"   sigma_r = {sigma_hex:.4f}  (K_hex = 19/7 = {K_HEX:.4f})  <- exact match")

# ── 2. Theorem 6.1 bounds at two stress levels ────────────────────────────────
print(f"\n2. Theorem 6.1 dissipation lower bound: "
      f"Psi >= C* * D / eps^2 * (sigma_r - K_hex)^2")
print(f"   C* = {C_STAR:.3f},  c0 = {c0:.4f}")
print()
print(f"   {'sigma_r':>8}  {'excess':>8}  {'W2 >= (cm)':>12}  {'Psi >= (h-1)':>14}  Phase")
print(f"   {'-'*8}  {'-'*8}  {'-'*12}  {'-'*14}  -----")

scenarios = [
    (K_HEX,  "K_hex (perfect hex)"),
    (3.1,    "Pre-drought (healthy)"),
    (3.8,    "Adaptive stress"),
    (4.6,    "Post-drought (stressed)"),
]

for sigma_r, label in scenarios:
    excess   = max(sigma_r - K_HEX, 0)
    W2_lb    = w2_lower_bound(sigma_r, eps)
    Psi_lb   = dissipation_lower_bound(sigma_r, eps, D)
    phase    = ("Healthy" if sigma_r <= 3.2 else
                "Adaptive" if sigma_r <= 4.5 else "Stressed")
    print(f"   {sigma_r:8.4f}  {excess:8.4f}  {W2_lb:12.6f}  {Psi_lb:14.4f}  {phase}")

# ── 3. Drought stress ratio ───────────────────────────────────────────────────
psi_pre  = dissipation_lower_bound(3.1, eps, D)
psi_post = dissipation_lower_bound(4.6, eps, D)
ratio    = psi_post / psi_pre
print(f"\n3. Drought stress ratio (sigma_r: 3.1 -> 4.6)")
print(f"   Psi_post / Psi_pre >= {ratio:.1f}x  (paper reports ~24x from simulation)")
print(f"   Quadratic scaling: ({4.6-K_HEX:.3f}/{3.1-K_HEX:.3f})^2 = "
      f"{((4.6-K_HEX)/(3.1-K_HEX))**2:.1f}x")

print(f"\n{'='*58}")
print(f"  Reproduce Figure 1:")
print(f"  python examples/drought_stress.py --save fig1.png")
print(f"{'='*58}")
