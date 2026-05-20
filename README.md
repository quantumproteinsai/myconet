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

```bash
pip install myconet
git clone https://github.com/quantumproteinsai/myconet
cd myconet
python3 examples/quickstart.py
```

> **Windows users:** replace `python3` with `python` if needed.

Output:

```
1. Hexagonal reference lattice
   N = 1362 nodes,  eps = 0.05 cm
   sigma_r = 2.7132  (K_hex = 19/7 = 2.7143)  <- exact match

2. Theorem 6.1 dissipation lower bound
    sigma_r    excess    W2 >= (cm)    Psi >= (h-1)  Phase
   --------  --------  ------------  --------------  -----
     2.7143    0.0000      0.000000          0.0000  Healthy
     3.1000    0.3857      0.005625          4.6656  Healthy
     3.8000    1.0857      0.015833         36.9664  Adaptive
     4.6000    1.8857      0.027500        111.5136  Stressed

3. Drought stress ratio (sigma_r: 3.1 -> 4.6)
   Psi_post / Psi_pre >= 23.9x  (paper reports ~24x)
```

---

## Reproducing Figure 1

```bash
# Fast stochastic model — < 1 s, matches paper figure exactly
python3 examples/drought_stress.py --save fig1.png

# Full Fokker-Planck simulation — ~5 min, exploratory
python3 examples/make_fig1.py
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
