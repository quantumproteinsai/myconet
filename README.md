# myconet

[![PyPI version](https://badge.fury.io/py/myconet.svg)](https://pypi.org/project/myconet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Python simulation framework for mycorrhizal network biophysics and general mathematical biology transport analysis, with Freiman–Villani thermodynamic diagnostics.**

---

## Overview

`myconet` implements a suite of mathematical tools originally developed for the analysis of mycorrhizal (fungal root) networks, grounded in Villani's hypocoercivity theory, the Freiman–Ruzsa structural theorem, and optimal transport (Wasserstein-2 / Sinkhorn). The library has since been applied as a unified transport diagnostic framework across a family of multiscale mathematical biology models for neurological and respiratory diseases.

**Core mathematical backbone:**

- **Wasserstein-2 distance** (Sinkhorn algorithm via POT) — measures divergence of a cell/network population distribution from a healthy reference
- **Relative entropy H(ρ|ρ∞)** — KL divergence from the steady state
- **Fisher information I(ρ|ρ∞)** — metabolic dissipation proxy
- **Freiman excess σ_r** — structural order relative to hexagonal reference
- **Villani dissipation lower bound Ψ_lb** — hypocoercivity-based convergence rate
- **Talagrand T₂ lower bound** — saturation ratio ρ_T = λ_hc·W₂²/(2H)
- **Fokker–Planck PDE solver** — mesoscale population transport

---

## Installation

```bash
pip install myconet
```

**Requirements:** `numpy`, `scipy`, `matplotlib`, `POT` (Python Optimal Transport), `scikit-learn`

---

## Core API

```python
from myconet.transport import wasserstein2, relative_entropy, fisher_information
from myconet.freiman  import dissipation_lower_bound, w2_lower_bound, freiman_excess
from myconet.network  import HyphalNetwork, hexagonal_lattice
from myconet.simulation import MycoNetSimulation, SimulationParams, run_ensemble
```

### Transport diagnostics

```python
import numpy as np
from myconet.transport import wasserstein2, relative_entropy, fisher_information

# nodes: (N, 2) array — current population distribution in R²
# ref  : (M, 2) array — healthy reference distribution
# eps  : float        — characteristic scale

nodes = np.random.rand(100, 2)          # current state (e.g. cell positions)
ref   = np.random.rand(100, 2) * 0.3   # healthy reference

W2     = wasserstein2(nodes, ref, eps=0.30)       # Sinkhorn W₂
H      = relative_entropy(nodes, ref, eps=0.30)   # KL divergence
I_fish = fisher_information(nodes, ref, eps=0.30) # Fisher information
```

### Freiman–Villani diagnostics

```python
from myconet.freiman import freiman_excess, w2_lower_bound, dissipation_lower_bound

sigma_r = freiman_excess(nodes, eps=0.30)              # structural excess
W2_lb   = w2_lower_bound(sigma_r, eps=0.30)            # Talagrand T₂ lower bound
Psi_lb  = dissipation_lower_bound(sigma_r, eps=0.30, D=0.05)  # Villani dissipation
```

---

## Applications

### Original application — Mycorrhizal network biophysics

`myconet` was developed for the companion paper:

> Mercier des Rochettes, B. (2026). *Geometric Efficiency Bounds for Mycorrhizal Networks: A Freiman–Villani Framework.* Journal of Mathematical Biology. `myconet` v1.0.0.

The core result: mycorrhizal hyphal networks self-organise toward the hexagonal lattice (K_HEX ≈ 0.6412) as the optimal transport configuration. The Freiman excess σ_r − K_HEX quantifies deviation from this optimum, and the HWI inequality W₂² ≤ (2/λ_hc)H bounds convergence.

---

### Biomedical applications (v1.1.0)

Starting from v1.1.0, `myconet` includes four complete multiscale disease models as worked examples. Each applies the Villani/HWI/Talagrand framework to a different biological system via a 2D embedding of the cellular state space.

**Embedding principle:** the 8-dimensional cellular state is projected onto two biologically meaningful planes — an inflammatory/pathological plane and a functional/repair plane — represented as Gaussian point clouds. The myconet transport functions then compute the divergence of this distribution from a healthy reference.

| Example | Disease | Inflammatory plane | Functional plane | Clinical readout |
|---|---|---|---|---|
| `ms_multiscale` | Multiple Sclerosis | (Teff, M1) | (OL, My) | EDSS, MRI lesion load |
| `ad_multiscale` | Alzheimer's Disease | (M1_micro, A1_astro) | (N_neu, N_syn) | MMSE, Aβ/tau PET |
| `pd_multiscale` | Parkinson's Disease | (DAQ, M1_micro) | (DA_neu, DA_level) | UPDRS-III, DAT-SPECT |
| `cf_multiscale` | Cystic Fibrosis | (Pa_biofilm, Neutrophil) | (CFTR_rescue, ASL_norm) | FEV1%, Pa index |

Each example is a self-contained four-scale model:
- **L0 Molecular:** Nakajima–Zwanzig non-Markovian memory kernels (Prony form)
- **L1 Aggregation:** Smoluchowski coagulation-fragmentation (disease-specific)
- **L2 Cellular:** ODE system + myconet Villani/HWI/Talagrand diagnostics
- **L3 Clinical:** Cusp catastrophe + validated clinical score

See `examples/` for full code and documentation.

---

## Mathematical background

### HWI inequality (Villani 2009)

For a probability measure ρ on ℝ² with λ_hc-log-Sobolev inequality:

$$W_2^2(\rho, \rho_\infty) \leq \frac{2}{\lambda_{hc}} H(\rho \,|\, \rho_\infty)$$

where W₂ is the Wasserstein-2 distance, H is the relative entropy (KL divergence), and λ_hc is the hypocoercivity rate (spectral gap of the linearised Fokker–Planck operator).

### Talagrand T₂ saturation ratio

$$\rho_T = \frac{\lambda_{hc} \cdot W_2^2}{2 H(\rho|\rho_\infty)} \in (0, 1]$$

ρ_T → 1 indicates the HWI bound is tight (population at thermodynamic equilibrium with the healthy reference). ρ_T < 1 indicates remaining distance to be covered — usable as a treatment response biomarker.

### Freiman excess

$$\sigma_r = \frac{|A + A|}{|A|} - K_{HEX}$$

where A is the point set of network nodes, |A+A| is the sumset cardinality, and K_HEX ≈ 0.6412 is the hexagonal packing constant. σ_r < 0 indicates over-ordered (pathological) polarisation; σ_r → 0 indicates convergence to the hexagonal optimum.

---

## Citation

If you use `myconet` in your research, please cite:

```bibtex
@software{myconet2026,
  author    = {Mercier des Rochettes, Bertrand},
  title     = {myconet: Python framework for mycorrhizal network biophysics
               and mathematical biology transport analysis},
  year      = {2026},
  version   = {1.1.0},
  publisher = {PyPI},
  url       = {https://pypi.org/project/myconet/},
  note      = {Quantum Proteins AI, Cergy-Pontoise, France}
}
```

For the biomedical applications, please also cite the corresponding JMB papers (references to be updated on acceptance):

- MS model: Mercier des Rochettes (2026, JMB, submitted)
- AD model: Mercier des Rochettes (2026, JMB, submitted)
- PD model: Mercier des Rochettes (2026, JMB, submitted)
- CF model: Mercier des Rochettes (2026, JMB, submitted)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## License

MIT — see [LICENSE](LICENSE).

**Author:** Bertrand Mercier des Rochettes  
**Institution:** Quantum Proteins AI, Cergy-Pontoise, France  
**Contact:** contact@quantum-proteins.ai  
**Website:** https://quantum-proteins.ai
