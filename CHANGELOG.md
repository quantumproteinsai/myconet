# Changelog

All notable changes to `myconet` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026

### Added

**Biomedical applications examples** (`examples/` directory):

Four complete multiscale disease simulation frameworks demonstrating the
generality of the myconet Villani/HWI/Talagrand transport diagnostic toolkit
beyond mycorrhizal networks. Each example is a self-contained four-scale
mathematical biology model targeting the *Journal of Mathematical Biology*.

- `examples/ms_multiscale/` — **Multiple Sclerosis**
  - Nakajima–Zwanzig non-Markovian cytokine memory kernels (TNF-α, IFN-γ, IL-10, TGF-β)
  - 8-variable cellular ODE (Teff/Treg, M1/M2, OPC/OL, myelin/axon integrity)
  - myconet embedding: inflammatory plane (Teff, M1) × repair plane (OL, My)
  - Cusp catastrophe for relapse-remission bifurcation
  - Clinical readouts: EDSS, MRI lesion load
  - Treatment: immunosuppression + remyelination dual therapy

- `examples/ad_multiscale/` — **Alzheimer's Disease**
  - NZ memory on Aβ nucleation lag + BACE1 quantum tunneling (KIE=1.88, IFP 1985)
  - Smoluchowski coagulation for Aβ oligomers/plaques + prion-like tau seeding
  - myconet embedding: neuroinflammatory plane (M1_micro, A1_astro) × neuronal plane (N_neu, N_syn)
  - Cusp catastrophe for MCI→AD bifurcation
  - Clinical readouts: MMSE, Aβ PET SUVr, tau PET
  - Treatment: anti-inflammatory + anti-amyloid + anti-tau triple therapy

- `examples/pd_multiscale/` — **Parkinson's Disease**
  - NZ memory on α-synuclein misfolding + dopamine-quinone (DAQ)/NRF1 redox axis
  - Quantum tunneling in Hsp70 chaperone H-transfer (KIE=1.30, IFP 1985)
  - Smoluchowski for α-syn oligomers → Lewy bodies + Braak staging propagation
  - myconet embedding: oxidative plane (DAQ, M1) × dopaminergic plane (DA_neu, DA)
  - Cusp catastrophe for motor ON/OFF bifurcation
  - Clinical readouts: UPDRS-III, DAT-SPECT, dyskinesia index
  - Treatment: levodopa + NRF1 activator + anti-α-syn triple therapy

- `examples/cf_multiscale/` — **Cystic Fibrosis (Mucoviscidosis)**
  - NZ memory on ΔF508 CFTR misfolding/ERAD + Hsp70 tunneling (KIE=1.45, IFP 1985)
  - Smoluchowski mucin cross-linking + ASL dehydration + PCL collapse
  - Pseudomonas aeruginosa planktonic→biofilm→mucoid phenotype switching
  - myconet embedding: infection plane (Pa_biofilm, Neutrophil) × epithelial plane (CFTR_rescue, ASL_norm)
  - Cusp catastrophe for acute exacerbation bifurcation
  - Clinical readouts: FEV1% predicted, Pseudomonas colonisation index, exacerbation risk
  - Treatment: Trikafta (elexacaftor/tezacaftor/ivacaftor) + tobramycin/azithromycin + dornase alfa

**Updated README:**
- Added Applications section describing the four disease examples
- Added mathematical background section (HWI inequality, Talagrand T₂, Freiman excess)
- Added embedding principle documentation
- Updated citation instructions

### Changed

- `__version__` bumped from `"1.0.0"` to `"1.1.0"` in `myconet/__init__.py`
- No breaking changes to any existing API

### Fixed

- No bug fixes in this release (core library unchanged)

---

## [1.0.6] — 2026 (PyPI release)

PyPI distribution release. Core library identical to 1.0.0.

---

## [1.0.0] — 2026

### Initial release

- `myconet.transport`: Fokker–Planck PDE solver, Wasserstein-2 (Sinkhorn/POT),
  Fisher information, relative entropy
- `myconet.freiman`: Freiman excess, Talagrand T₂ lower bound, Villani dissipation bound
- `myconet.network`: HyphalNetwork class, hexagonal lattice generator
- `myconet.simulation`: MycoNetSimulation, SimulationParams, run_ensemble

Companion to: Mercier des Rochettes, B. (2026). *Geometric Efficiency Bounds for
Mycorrhizal Networks: A Freiman–Villani Framework.* Journal of Mathematical Biology.
