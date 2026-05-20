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
from myconet import MycoNetSimulation

sim     = MycoNetSimulation(seed=42)
results = sim.run(T=120, drought_onset=48)
results.summary()
results.plot()
```

---

## Reproducing Figure 1

```bash
# Single run (~3 min)
python examples/drought_stress.py --save fig1.png

# 10-run ensemble — matches paper exactly (~25 min)
python examples/drought_stress.py --ensemble --save fig1.png
```

**On a VPS or remote server** (runs in background):

```bash
pip install myconet
git clone https://github.com/quantumproteinsai/myconet
cd myconet
pip install -e ".[dev]"

screen -S myconet
python examples/drought_stress.py --ensemble --save fig1.png
# Ctrl+A then D to detach — reconnect with: screen -r myconet
```

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
  version = {1.0.0}
}
```

---

## License

MIT © 2026 Bertrand Mercier des Rochettes / Quantum Proteins AI
