# Parkinson's Disease — Multiscale Model

**`myconet` example: `pd_multiscale`**

A four-scale mathematical framework for Parkinson's disease integrating
NZ non-Markovian α-synuclein misfolding memory, dopamine-quinone (DAQ)/NRF1
redox dynamics, Smoluchowski-Lewy body aggregation, Braak staging propagation,
Villani hypocoercivity, and cusp catastrophe for motor ON/OFF bifurcation,
with myconet transport diagnostics.

---

## Biological motivation

PD pathology unfolds across three coupled axes:

```
Native α-syn → [misfolding, NZ memory lag] → misfolded α-syn
                 ↑ KIE=1.30 (Hsp70 ATPase H-transfer, IFP 1985)
                 ↑ DAQ oxidative modification accelerates misfolding

Misfolded α-syn → [Smoluchowski] → oligomers → Lewy bodies
                                        ↓ prion-like Braak propagation

Lewy body burden + DAQ + M1 microglia → DA neuron death (SNpc)
                                              ↓
Striatal dopamine deficit → D1/D2 imbalance → motor symptoms (UPDRS-III)
                                              ↓
ON/OFF motor fluctuations → cusp catastrophe bifurcation
```

**DAQ/NRF1 redox axis** (from Paper 2, Physical Biology framework):
- Dopamine auto-oxidation → DAQ (dopamine-quinone)
- DAQ covalently modifies α-syn → accelerates misfolding
- NRF1 activates ARE → clears DAQ (therapeutic target)
- Mitochondrial dysfunction (age-dependent) → DAQ overproduction

---

## Mathematical structure

### L0 — Molecular-quantum: NZ α-syn + DAQ/NRF1

```
ds_mis/dt = k_mis(DAQ,age)·s_nat − δ_mis·s_mis + Σ m_k − clearance
dm_k/dt   = γ_k · s_mis − m_k / τ_k    (τ = 3, 8, 15 days)

k_mis = k₀ · age_factor · (1 + K_DAQ·DAQ) · [1 + f_tunnel/KIE_{H/D}]
KIE_{H/D} = 1.30    (Hsp70 ATPase H-transfer, IFP 1985)

dDAQ/dt  = k_DAQ · DA_level · age_factor − K_NRF1 · NRF1 · DAQ − δ·DAQ
dNRF1/dt = k_base + k_act · DAQ/(DAQ+0.3) + u_nrf1·0.40 − δ·NRF1
```

### L1 — Aggregation: Smoluchowski-Lewy + Braak

Smoluchowski for α-syn aggregates (N_max=10):
nucleation source: f₁ = k_nuc · s²_mis

Braak staging (continuous approximation):
```
dBraak/dt = k_Braak · LB · (6 − Braak) / 6
LB = Σ_{i≥4} i · c_i    (Lewy body index, size-weighted)
```

DA neuron loss:
```
dDA_neu/dt = −(k_LB·LB + k_DAQ·DAQ + k_M1·M1) · DA_neu
```

### L2 — Cellular: nigrostriatal circuit + myconet

6-variable ODE: [DA_level, D1_rec, D2_rec, M1_micro, M2_micro, Astrocytes]

**myconet 2D embedding:**
- Oxidative plane: (DAQ, M1_microglia) — neurotoxicity axis
- Dopaminergic plane: (DA_neurons, DA_level) — functional axis

### L3 — Clinical: Cusp motor ON/OFF + UPDRS

```
u = |D1 − D2| − 0.4·DA     (D1/D2 receptor imbalance − dopamine)
v = DA_neu + 0.4·DA          (nigrostriatal reserve)

UPDRS-III = 28·dopamine_deficit + 24·neuron_loss + 14·receptor_imbalance + 14·Braak_burden
```

Dyskinesia index: DYS = max(0, u_lev − 0.35) × (1−DA) × D1 × 2

---

## Running

```bash
pip install numpy scipy matplotlib myconet
python pd_multiscale.py
```

Runtime: ~25 seconds (4 scenarios × 1825 days, dt=3 days)

---

## Scenarios

| Scenario | Levodopa | NRF1 activator | Anti-α-syn | Onset |
|---|---|---|---|---|
| Early PD — Untreated | 0.0 | 0.0 | 0.0 | — |
| Early PD — Triple Therapy | 0.60 | 0.50 | 0.30 | Day 180 |
| Advanced PD — Untreated | 0.0 | 0.0 | 0.0 | — |
| Advanced PD — Triple Therapy | 0.70 | 0.55 | 0.40 | Day 180 |

---

## Expected results (year 5)

| Metric | Advanced PD Untreated | Advanced PD Triple |
|---|---|---|
| UPDRS-III | ~44 | ~23 (ΔUPDRS = +21) |
| DA neurons | ~0.41 | ~0.41 |
| Braak stage | ~3.3 | ~3.2 |
| W₂ (myconet) | ~1.1 | ~1.7 |
| Dyskinesia | — | 0.0 |

**Key result:** NRF1 activation reduces DAQ, preventing D1 receptor
hypersensitisation, thereby eliminating levodopa-induced dyskinesia —
one of the most important unmet needs in PD pharmacology.

---

## Connection to Physical Biology Paper 2

The DAQ/NRF1 terms in the L0 layer are directly taken from the companion
manuscript: Mercier des Rochettes, B. (2026). *α-Synuclein Aggregation in
Parkinson's Disease with Dopamine-Quinone and NRF1 Terms.* Physical Biology
(submitted). The two papers share the L0 molecular layer and are
complementary: the Physical Biology paper focuses on the molecular-scale
dynamics; this JMB paper extends to the clinical-scale via Villani/myconet.

---

## Citation

```bibtex
@article{mercier2026pd,
  author  = {Mercier des Rochettes, Bertrand},
  title   = {A Non-Markovian Multiscale Framework for {Parkinson's} Disease:
             {NZ} Memory, {DAQ}/{NRF1} Redox, {Smoluchowski}--{Lewy} Aggregation,
             and Cusp Motor {ON}/{OFF} Bifurcation},
  journal = {Journal of Mathematical Biology},
  year    = {2026},
  note    = {Submitted. myconet v1.1.0.}
}
```
