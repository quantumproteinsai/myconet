# Alzheimer's Disease — Multiscale Model

**`myconet` example: `ad_multiscale`**

A four-scale mathematical framework for Alzheimer's disease integrating
NZ non-Markovian Aβ nucleation memory, Smoluchowski coagulation, prion-like
tau propagation, Villani hypocoercivity, and cusp catastrophe for MCI→AD
transitions, with myconet transport diagnostics.

Companion paper: Mercier des Rochettes, B. (2026). *A Non-Markovian Multiscale
Framework for Alzheimer's Disease.* Journal of Mathematical Biology (submitted).

---

## Biological motivation

AD pathology follows a cascade across three interacting axes:

```
APP → [BACE1 cleavage] → Aβ monomer → oligomers → plaques (Smoluchowski)
                ↑ KIE=1.88 tunneling correction (IFP 1985)
                ↑ NZ memory on nucleation lag (τ=5,10,20 days)

Aβ oligomers → tau seeding → NFT propagation (prion-like, Braak staging)
                                    ↓
Neuroinflammation (M1/M2 microglia, A1/A2 astrocytes)
                                    ↓
Synaptic loss → MMSE decline → MCI → dementia (cusp bifurcation)
```

The **type-3-diabetes axis**: insulin resistance upregulates BACE1,
modelled as a slow drift ins(t) = 0.10 + 0.15·tanh(t/1200).

---

## Mathematical structure

### L0 — Molecular-quantum: NZ memory + BACE1 tunneling

```
dc_APP/dt = P_APP − k_BACE(ins, t) · c_APP − δ_APP · c_APP
dc_mon/dt = k_BACE · c_APP − k_nuc · c_mon − δ_mon · c_mon + Σ m_k
dm_k/dt   = γ_k · c_mon − m_k / τ_k        (τ = 5, 10, 20 days)

k_BACE = k₀ · (1 + 0.4·ins) · α(t) · [1 + f_tunnel/KIE_{H/D}]
KIE_{H/D} = 1.88    (IFP 1985 experimental value)
```

### L1 — Aggregation: Smoluchowski + prion-like tau

Truncated Smoluchowski (N_max = 12):
```
dc_i/dt = ½ Σ_{j=1}^{i-1} k_{j,i-j} c_j c_{i-j}
           − c_i Σ_j k_{i,j} c_j − δ_i c_i + source
k_{i,j} = k₀ (i^{1/3} + j^{1/3})²   (diffusion-limited)
```

Tau phosphorylation (prion-like seeding):
```
dP_τ/dt = P₀ + k_seed · c_olig · (1−P_τ) + k_spread · P_τ · (1−P_τ) − k_clear · P_τ
```

### L2 — Cellular: neuroinflammation + myconet

6-variable ODE: [M1, M2, A1, A2, N_syn, BDNF]

**myconet 2D embedding:**
- Neuroinflammatory plane: (M1_micro, A1_astro)
- Neuronal plane: (N_neurons, N_synaptic)

Diagnostics computed every 90 days (Sinkhorn stride) via:
```python
from myconet.transport import wasserstein2, relative_entropy, fisher_information
from myconet.freiman  import dissipation_lower_bound, w2_lower_bound, freiman_excess
```

### L3 — Clinical: Cusp catastrophe + MMSE

```
u = 0.4·(1−N_syn) + 0.3·P_τ     (cognitive burden)
v = 0.6·N_neu + 0.4·BDNF         (neuroprotective reserve)
MMSE = 12·N_syn + 10·N_neu + 5·BDNF − 8·P_τ − 3·(1−N_neu)·(1−N_syn)
```

---

## Running

```bash
pip install numpy scipy matplotlib myconet
python ad_multiscale.py
```

Runtime: ~30 seconds (4 scenarios × 1825 days, dt=3 days)

Produces 7 figures in `figures/`:

| Figure | Content |
|---|---|
| fig1_clinical.png | MMSE, Aβ PET, Tau PET trajectories |
| fig2_smoluchowski.png | NZ molecular + Smoluchowski aggregation |
| fig3_villani.png | myconet Villani/HWI/Talagrand diagnostics |
| fig4_cusp.png | Cusp catastrophe MCI→AD bifurcation |
| fig5_cellular.png | M1/M2/A1/A2/N_syn/BDNF dynamics |
| fig6_summary.png | 4-scale integration summary (12 panels) |
| fig7_dual_w2.png | Therapeutic W₂ paradox |

---

## Scenarios

| Scenario | Anti-inflam. | Anti-Aβ | Anti-tau | Onset |
|---|---|---|---|---|
| MCI — Untreated | 0.0 | 0.0 | 0.0 | — |
| MCI — Triple Therapy | 0.50 | 0.60 | 0.40 | Day 365 |
| Late-AD — Untreated | 0.0 | 0.0 | 0.0 | — |
| Late-AD — Triple Therapy | 0.40 | 0.70 | 0.55 | Day 365 |

---

## Validated results (year 5)

| Metric | MCI Unt. | MCI Triple | Late-AD Unt. | Late-AD Triple |
|---|---|---|---|---|
| MMSE | 6.3 | 23.5 (Δ=+17.1) | 1.7 | 14.7 (Δ=+13.0) |
| P_tau | 0.285 | 0.002 | 0.288 | 0.002 |
| N_neurons | 0.538 | 0.851 | 0.325 | 0.548 |
| W₂ (myconet) | 1.052 | 0.152 | 0.585 | 1.088 |
| σ_r (Freiman) | −0.284 | −0.036 | −0.219 | −0.058 |

**MMSE decline rate:** 2.9 pts/yr untreated MCI — consistent with ADNI cohort data.
**ΔFEV1:** Triple therapy MMSE gain of +17.1 represents theoretical upper bound
for simultaneous complete multi-target intervention.

**The therapeutic W₂ paradox:** W₂ decreases under treatment in MCI (population
converging to healthy reference) but increases in Late-AD (population escaping
the deeply established pathological attractor). This staging-dependent signature
is consistent across MS, AD, PD, and CF in the myconet framework.

---

## Position vs. literature

This model extends Putra et al. (JMB 2025) — network aggregation AD — by adding:
1. NZ non-Markovian nucleation memory (explains BACE1 inhibitor failures)
2. Villani transport diagnostics on neuroinflammatory populations
3. Cusp catastrophe for the MCI→AD tipping point
4. BACE1 quantum tunneling (KIE=1.88, experimentally anchored)
5. Therapeutic W₂ paradox as a novel optimal transport biomarker

---

## Citation

```bibtex
@article{mercier2026ad,
  author  = {Mercier des Rochettes, Bertrand},
  title   = {A Non-Markovian Multiscale Framework for {Alzheimer's} Disease:
             {Smoluchowski} Aggregation, Villani Hypocoercivity, and Cusp
             Catastrophe for {MCI}--{AD} Transitions},
  journal = {Journal of Mathematical Biology},
  year    = {2026},
  note    = {Submitted. myconet v1.1.1.}
}
```

