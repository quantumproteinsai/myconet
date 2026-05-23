# myconet Examples

This directory contains four biomedical application examples demonstrating
the generality of the `myconet` Villani/HWI/Talagrand transport framework.

Each example is a complete, self-contained multiscale mathematical biology
simulation targeting the *Journal of Mathematical Biology* (JMB).

---

## Installation

All examples require only:

```bash
pip install numpy scipy matplotlib myconet
```

No additional dependencies beyond the core myconet install.

---

## Examples

| Directory | Disease | Key novelty | Clinical readout |
|---|---|---|---|
| `ms_multiscale/` | Multiple Sclerosis | NZ cytokine memory + Villani | EDSS, MRI |
| `ad_multiscale/` | Alzheimer's Disease | NZ nucleation + Smoluchowski | MMSE, PET |
| `pd_multiscale/` | Parkinson's Disease | NZ α-syn + DAQ/NRF1 redox | UPDRS-III, DAT-SPECT |
| `cf_multiscale/` | Cystic Fibrosis | NZ ΔF508 CFTR + mucin Smoluchowski | FEV1%, Pa index |

---

## Shared architecture

All four examples follow the same four-scale structure:

```
L0  Molecular-quantum : Nakajima–Zwanzig non-Markovian memory kernels (Prony)
                        + quantum tunneling correction (IFP 1985 KIE data)
L1  Aggregation       : Smoluchowski coagulation-fragmentation
                        (disease-specific: cytokines / Aβ / α-syn / mucin)
L2  Cellular          : ODE system + myconet Villani/HWI/Talagrand diagnostics
                        (2D embedding of cellular state → transport functions)
L3  Clinical          : Cusp catastrophe + validated clinical score
```

### The 2D embedding principle

The key interface between each disease model and `myconet` is the 2D embedding:
the high-dimensional cellular state is projected onto two biologically
meaningful planes, represented as Gaussian point clouds, and passed to
`myconet.transport.wasserstein2()` and related functions.

```
myconet functions called (identical across all four examples):
  transport.wasserstein2()              → Sinkhorn W₂ distance
  transport.relative_entropy()          → KL divergence H(ρ|ρ∞)
  transport.fisher_information()        → Fisher information I(ρ|ρ∞)
  freiman.freiman_excess()              → Freiman structural excess σ_r
  freiman.w2_lower_bound()              → Talagrand T₂ lower bound
  freiman.dissipation_lower_bound()     → Villani dissipation bound Ψ_lb
```

### The IFP 1985 connection

Each disease example incorporates a quantum tunneling correction to a key
molecular rate constant, anchored in experimental KIE(H/D) data from:

> Mercier des Rochettes, B. (1985). *Hydrogen transfer in conformationally
> constrained bicyclic systems.* Doctoral thesis, IFP, France.

| Disease | Process | KIE(H/D) |
|---|---|---|
| AD | BACE1 β-secretase H-transfer | 1.88 |
| PD | Hsp70 ATPase H-transfer (α-syn) | 1.30 |
| CF | Hsp70 ATPase H-transfer (ΔF508 CFTR) | 1.45 |

---

## Running the examples

Each example produces 6 paper-quality PNG figures and a summary statistics
printout. Runtime is approximately 15–30 seconds per example on a standard
laptop (the Sinkhorn computation is the rate-limiting step, called every
90 simulation days).

```bash
# Multiple Sclerosis
cd ms_multiscale && python ms_myconet.py

# Alzheimer's Disease
cd ad_multiscale && python ad_multiscale.py

# Parkinson's Disease
cd pd_multiscale && python pd_multiscale.py

# Cystic Fibrosis
cd cf_multiscale && python cf_multiscale.py
```

Expected figure outputs per example:

| Figure | Content |
|---|---|
| `fig1_clinical.png` | Clinical trajectories (2 phenotypes × 2 treatments) |
| `fig2_*.png` | Molecular/aggregation layer dynamics |
| `fig3_villani.png` | myconet diagnostics (W₂, H, I, Ψ_lb, σ_r, ρ_T) |
| `fig4_cusp.png` | Cusp catastrophe 3D surface + control-space trajectories |
| `fig5_*.png` | Cellular layer dynamics |
| `fig6_summary.png` | 4-scale integration summary (12-panel overview) |

---

## Citation

If you use these examples in your research, please cite:

```bibtex
@software{myconet2026,
  author  = {Mercier des Rochettes, Bertrand},
  title   = {myconet v1.1.0},
  year    = {2026},
  url     = {https://pypi.org/project/myconet/},
}
```

And the corresponding disease paper(s) — see each example's README for
the specific JMB citation.
