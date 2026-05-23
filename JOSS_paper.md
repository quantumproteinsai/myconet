---
title: 'myconet: A Python Framework for Optimal Transport and Villani Hypocoercivity Diagnostics in Mathematical Biology'
tags:
  - Python
  - optimal transport
  - Wasserstein distance
  - Villani hypocoercivity
  - Freiman-Ruzsa theorem
  - mathematical biology
  - mycorrhizal networks
  - multiple sclerosis
  - Alzheimer disease
  - Parkinson disease
  - cystic fibrosis
authors:
  - name: Bertrand Mercier des Rochettes
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Quantum Proteins AI, Cergy-Pontoise, France
    index: 1
date: 2026
bibliography: paper.bib
---

# Summary

`myconet` is a Python library providing a suite of mathematical tools for
analysing probability distributions arising in biological network and cell
population dynamics, grounded in Villani's hypocoercivity theory
[@villani2009optimal], the Freiman–Ruzsa structural theorem [@freiman1973],
and Wasserstein-2 optimal transport [@villani2003topics]. Originally developed
for mycorrhizal (fungal root) network biophysics, the library has been
generalised as a unified transport diagnostic framework applicable to any
biological system whose state can be represented as a probability measure
evolving toward a healthy reference distribution.

The core contribution is a set of thermodynamic diagnostics that quantify,
at each timestep of a multiscale simulation:

- **W₂** (Sinkhorn–Wasserstein-2 distance): the optimal transport cost from
  current state to healthy reference [@peyre2019computational]
- **H** (relative entropy): KL divergence from the healthy steady state
- **I** (Fisher information): metabolic dissipation proxy
- **σ_r** (Freiman excess): structural order relative to the hexagonal
  reference lattice
- **Ψ_lb** (Villani dissipation lower bound): hypocoercivity convergence rate
- **ρ_T** (Talagrand T₂ saturation ratio): whether the HWI inequality is
  saturated, usable as a treatment response biomarker

These diagnostics are connected by the HWI inequality [@villani2009optimal]:

$$W_2^2(\rho_t, \rho_\infty) \leq \frac{2}{\lambda_{hc}} H(\rho_t \,|\, \rho_\infty)$$

where λ_hc is the hypocoercivity rate (spectral gap of the linearised
Fokker–Planck operator), and ρ_∞ is the healthy reference measure.

# Statement of need

Mathematical biology increasingly requires tools that bridge molecular-scale
dynamics, cellular-scale population behaviour, and clinical-scale outcomes
[@keener2008mathematical]. Existing software packages address specific
subproblems (ODE integration, PDE solvers, network analysis) but no unified
framework exists for computing Villani-type transport diagnostics on the
cell-population distributions arising from these systems.

`myconet` fills this gap. It provides:

1. A **Fokker–Planck PDE solver** for mesoscale population transport
2. **Sinkhorn–Wasserstein-2** computation via the Python Optimal Transport
   library [@flamary2021pot], with nearest-neighbour fallback for robustness
3. **Fisher information** and **relative entropy** estimators via kernel
   density estimation on a 2D grid
4. **Freiman excess** and **Villani dissipation bound** from the
   Freiman–Ruzsa structural theorem [@freiman1973], applied to point clouds

The **2D embedding principle** enables these spatial tools to be applied to
high-dimensional biological state vectors: the relevant biological variables
are projected onto two clinically meaningful planes (e.g. inflammatory vs.
repair, or infection vs. function), represented as Gaussian point clouds,
and passed to the myconet functions. This approach is mathematically justified
because the HWI inequality holds in any marginal subspace of the full
distribution.

# Original application — Mycorrhizal networks

`myconet` was developed for the analysis of mycorrhizal hyphal networks,
modelled as spatial point processes on ℝ². The central result is that
optimal mycorrhizal networks self-organise toward the hexagonal lattice
(Freiman constant K_HEX ≈ 0.6412) as the minimum-cost transport configuration.
The Freiman excess σ_r − K_HEX quantifies deviation from this optimum, and
the Villani dissipation bound Ψ_lb provides a thermodynamic lower bound on
the metabolic cost of maintaining the network [@mercier2026myco].

The library includes a complete simulation framework (`MycoNetSimulation`,
`SimulationParams`, `run_ensemble`) for this original application.

# Biomedical applications

Starting from v1.1.0, `myconet` includes four worked examples demonstrating
the framework's generality across disease biology. Each example is a
four-scale multiscale model coupling:

- **L0**: Nakajima–Zwanzig non-Markovian memory kernels [@nakajima1958;
  @zwanzig1960] (Prony form) for molecular-scale kinetics with memory
- **L1**: Smoluchowski coagulation-fragmentation equations
  [@smoluchowski1917] for protein/polymer aggregation
- **L2**: ODE system for cellular/microbial population dynamics, with
  myconet transport diagnostics via 2D embedding
- **L3**: Cusp catastrophe [@zeeman1977] for clinical bifurcation analysis

## Multiple sclerosis [@mercier2026ms]

NZ memory on cytokine cross-regulation (TNF-α, IFN-γ, IL-10, TGF-β);
8-variable cellular ODE (Teff/Treg, M1/M2, OPC/OL, myelin/axon); cusp
catastrophe for relapse-remission bifurcation; EDSS and MRI lesion readouts.

## Alzheimer's disease [@mercier2026ad]

NZ memory on Aβ nucleation lag; BACE1 quantum tunneling correction
(KIE_{H/D} = 1.88, anchored in [@mercier1985]); truncated Smoluchowski
(N=12) for Aβ oligomers/plaques; prion-like tau seeding; cusp catastrophe
for MCI→AD transition; MMSE and PET biomarker readouts.

## Parkinson's disease [@mercier2026pd]

NZ memory on α-synuclein misfolding; dopamine-quinone/NRF1 redox axis;
Hsp70 tunneling correction (KIE_{H/D} = 1.30, [@mercier1985]); Smoluchowski-
Lewy body aggregation with Braak staging propagation; cusp catastrophe for
motor ON/OFF bifurcation; UPDRS-III and DAT-SPECT readouts.

## Cystic fibrosis [@mercier2026cf]

NZ memory on ΔF508 CFTR misfolding (ERAD pathway); Hsp70 tunneling
correction (KIE_{H/D} = 1.45, [@mercier1985]); Smoluchowski mucin
cross-linking with concentration-dependent kernel and ASL dehydration
dynamics; Pseudomonas aeruginosa phenotype switching (planktonic→biofilm→
mucoid); cusp catastrophe for exacerbation onset; FEV1% and Pa colonisation
readouts. Models the post-Trikafta clinical landscape where CFTR rescue and
residual Pseudomonas infection coexist.

# The IFP 1985 experimental anchor

A unifying feature of the three protein-misfolding examples (AD, PD, CF)
is the quantum tunneling correction applied to a key H-transfer rate constant.
These corrections are anchored in experimental kinetic isotope effect (KIE)
measurements from [@mercier1985], which demonstrated geometry-dependent
suppression of hydrogen tunneling in conformationally constrained bicyclic
systems — an early experimental demonstration of what is now recognised as
a foundational result in quantum biology. The 40-year arc from that 1985
experimental observation to its application in multiscale disease models
via myconet represents an unusual continuity of research programme.

# Implementation

`myconet` is implemented in pure Python with NumPy [@harris2020array] and
SciPy [@virtanen2020scipy] dependencies. The Sinkhorn–Wasserstein-2
computation uses the Python Optimal Transport library [@flamary2021pot].
The Fokker–Planck solver uses finite differences (central for diffusion,
upwind for advection). All biomedical examples use inline RK4 integration
for speed, replacing scipy.integrate.solve_ivp calls to avoid per-step
overhead in long simulations (1825 days × 4 scenarios ≈ 2400 integration
steps each).

# Acknowledgements

The author thanks the Carmelite community of Paris for the contemplative
environment in which much of this work was developed, and acknowledges
the foundational role of the IFP (Institut Français du Pétrole) doctoral
programme (1985) in establishing the experimental basis for the quantum
tunneling corrections.

# References
