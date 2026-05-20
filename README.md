# MycoNet

**Python simulation framework for mycorrhizal network biophysics with Freiman–Villani thermodynamic analysis.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-myconet-blue)](https://pypi.org/project/myconet)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b)](https://arxiv.org/abs/2026.XXXXX)

---

## Overview

MycoNet implements the simulation and computational validation described in:

> Mercier des Rochettes, B. (2026). *MycoNet: A Python Framework for Mycorrhizal Network
> Biophysics.* arXiv:2026.XXXXX [q-bio.QM]

Companion theory paper:

> Mercier des Rochettes, B. (2026). *Geometric Efficiency Bounds for Mycorrhizal Networks:
> A Freiman–Villani Framework.* Journal of Mathematical Biology. arXiv:2026.YYYYY

The central result (Theorem 6.1): a mycorrhizal network with local Freiman index σ_r(Γ) satisfies

    Ψ(Γ) ≥ C* · D / ε² · (σ_r(Γ) − K_hex)²

where K_hex = 19/7 ≈ 2.714, C* ≈ 21.8, ε is mean hyphal spacing, D is diffusivity.

---

## Installation

```bash
pip install myconet
```

From source:

```bash
git clone https://github.com/quantumproteinsai/myconet
cd myconet
pip install -e ".[dev]"
```

---

## Quick start

```python
from myconet import local_freiman_index, hexagonal_lattice, K_HEX, C_STAR, c0
import numpy as np

# Compute Freiman index on a hexagonal lattice
nodes = hexagonal_lattice(eps=0.05, domain_size=2.0)
sigma, per_node = local_freiman_index(nodes, eps=0.05, k=7)
print(f"sigma_r = {sigma:.4f}  (K_hex = {K_HEX:.4f})")
# → sigma_r = 2.7137  (K_hex = 2.7143)
```

```bash
# Reproduce Figure 1 of the paper (< 1 s)
pip install myconet
git clone https://github.com/quantumproteinsai/myconet
cd myconet
python examples/drought_stress.py --save fig1.png
```

---

## Reproducing Figure 1

```bash
# Fast stochastic model — < 1 s, matches paper figure exactly
python examples/drought_stress.py --save fig1.png

# Full Fokker-Planck simulation — ~5 min, exploratory
python examples/make_fig1.py
```

> **Note:** `drought_stress.py` implements the stochastic model parameterised to match
> the Freiman–Villani theoretical predictions (σ_r: 3.1 → 4.6, C_sim ≈ 20.9).
> `MycoNetSimulation` runs the full coupled PDE + network simulation for research use.

---

## Key modules

| Module | Content |
|---|---|
| `myconet.freiman` | Local Freiman index via k-NN + hex-integer FFT Minkowski sum |
| `myconet.network` | HyphalNetwork, hexagonal lattice generation, drift field |
| `myconet.transport` | Fokker–Planck solver, Wasserstein W₂ (log-Sinkhorn/POT), Fisher info |
| `myconet.simulation` | MycoNetSimulation, SimulationParams, ensemble runner |

**Theoretical constants (all exact):**

| Constant | Value | Source |
|---|---|---|
| `K_HEX` | 19/7 ≈ 2.714 | Lemma 4.1: hexagonal local doubling constant |
| `C_STAR` | 256·49/576 ≈ 21.8 | Theorem 6.1: dissipation bound constant |
| `c0` | 7/24 ≈ 0.292 | Proposition 4.4a: perturbative Freiman–Wasserstein constant |
| `c0_TV` | √7/24 ≈ 0.110 | Proposition 4.4b: universal Tao–Vu constant |

---

## Citation

```bibtex
@software{myconet2026,
  author  = {Mercier des Rochettes, Bertrand},
  title   = {{MycoNet}: Python simulation framework for mycorrhizal network biophysics},
  year    = {2026},
  url     = {https://github.com/quantumproteinsai/myconet},
  version = {1.0.6}
}
```

---

## License

MIT © 2026 Bertrand Mercier des Rochettes / Quantum Proteins AI
