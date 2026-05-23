# Alzheimer's Disease — Multiscale Model

**`myconet` example: `ad_multiscale`**

A four-scale mathematical framework for Alzheimer's disease integrating
NZ non-Markovian Aβ nucleation memory, Smoluchowski coagulation, prion-like
tau propagation, Villani hypocoercivity, and cusp catastrophe for MCI→AD
transitions, with myconet transport diagnostics.

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

---

## Scenarios

| Scenario | Anti-inflam. | Anti-Aβ | Anti-tau | Onset |
|---|---|---|---|---|
| MCI — Untreated | 0.0 | 0.0 | 0.0 | — |
| MCI — Triple Therapy | 0.50 | 0.60 | 0.40 | Day 365 |
| Late-AD — Untreated | 0.0 | 0.0 | 0.0 | — |
| Late-AD — Triple Therapy | 0.40 | 0.70 | 0.55 | Day 365 |

---

## Expected results (year 5)

| Metric | MCI Untreated | MCI Triple Therapy |
|---|---|---|
| MMSE | ~6 | ~23 (ΔMMSE = +17) |
| P_tau | ~0.28 | ~0.002 |
| N_neurons | ~0.54 | ~0.85 |
| W₂ (myconet) | ~1.05 | ~0.15 |

---

## Position vs. literature

This model extends Putra et al. (JMB 2025) — network aggregation AD — by adding:
1. NZ non-Markovian nucleation memory (explains BACE1 inhibitor failures)
2. Villani transport diagnostics on neuroinflammatory populations
3. Cusp catastrophe for the MCI→AD tipping point
4. BACE1 quantum tunneling (KIE=1.88, experimentally anchored)

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
  note    = {Submitted. myconet v1.1.0.}
}
```
