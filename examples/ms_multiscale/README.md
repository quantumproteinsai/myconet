# Cystic Fibrosis (Mucoviscidosis) — Multiscale Model

**`myconet` example: `cf_multiscale`**

A four-scale mathematical framework for cystic fibrosis (ΔF508 genotype)
integrating NZ non-Markovian CFTR misfolding memory, Smoluchowski mucin
cross-linking with ASL dehydration, Pseudomonas aeruginosa phenotype switching,
Villani hypocoercivity, and cusp catastrophe for acute exacerbation onset,
with myconet transport diagnostics.

---

## Biological motivation

CF pathology involves three coupled axes unique among the myconet examples:

```
ΔF508 CFTR → [misfolding, NZ memory lag] → ER retention → ERAD
              ↑ KIE=1.45 (Hsp70 ATPase H-transfer, IFP 1985)
              ↑ geometric constraint of ΔF508 β-hairpin (bicyclic analogy)

ΔF508 CFTR misfolding → reduced Cl⁻/water secretion → ASL dehydration
                                        ↓
Mucin hypersecretion → [Smoluchowski cross-linking] → viscoelastic gel
                                        ↓
PCL collapse → mucociliary clearance failure → Pseudomonas colonisation
                                        ↓
Planktonic Pa → Biofilm Pa → Mucoid Pa (chronic, antibiotic-resistant)
                                        ↓
Neutrophil recruitment → IL-8 storm → epithelial damage → FEV1 decline
                                        ↓
Acute exacerbation (cusp catastrophe bifurcation)
```

**Post-Trikafta reality modelled explicitly:**
Trikafta (elexacaftor/tezacaftor/ivacaftor) rescues CFTR trafficking (L0),
restores ASL height (L1), but chronic Pseudomonas mucoid biofilm persists (L2)
— requiring continued antibiotic strategy even after CFTR correction.

---

## Mathematical structure

### L0 — Molecular-quantum: NZ ΔF508 CFTR misfolding + ERAD

```
dc_mis/dt = K_MIS·c_wt − k_chap·c_mis − K_ERAD·c_mis + Σ m_k − clearance
dm_k/dt   = γ_k · c_mis − m_k / τ_k    (τ = 4, 12, 30 days)

k_chap = k₀ · (1 + u_tri·1.8) · [1 + f_tunnel/KIE_{H/D}]
KIE_{H/D} = 1.45    (Hsp70 ATPase H-transfer in ΔF508 β-hairpin, IFP 1985)

CFTR_rescue rate: k_chap·c_mis·K_RESCUE + u_tri·0.25·(1−CFTR_rescue)
```

Memory timescales reflect Hsp70 (τ=4d), Hsp90 (τ=12d), and CHIP/co-chaperone
(τ=30d) engagement cycles in the ERAD decision pathway.

### L1 — Aggregation: Smoluchowski mucin + ASL dynamics

Concentration-dependent coagulation kernel:
```
k_{i,j}(t) = k₀ · (i^{1/3} + j^{1/3})² · C_muc / C_muc0
C_muc = Σ_i i·c_i / h    (mucin concentration = volume-normalised)
```

Viscosity (power law, α=2.5 for CF mucus):
```
η = η₀ · (C_muc / C_muc0)^α
```

ASL height dynamics:
```
dh/dt = J_ion(CFTR_rescue) − v_MCC(η, h) · h − evaporation
v_MCC = 0.05 · h/(h + h_PCL) / η^0.8    (mucociliary clearance)
h_PCL = 7 μm    (PCL collapse threshold)
```

### L2 — Cellular: Pseudomonas switching + myconet

6-variable ODE: [Pa_planktonic, Pa_biofilm, Pa_mucoid, Neutrophil, IL-8, Epithelial_integrity]

Planktonic→biofilm switch (ASL-dependent):
```
switch_rate = 0.08 · max(0, 1 − h/0.06)    (more switching when ASL dehydrated)
```

**myconet 2D embedding:**
- Infection plane: (Pa_biofilm, Neutrophil) — infection/inflammation axis
- Epithelial plane: (CFTR_rescue, ASL_norm) — functional/hydration axis

### L3 — Clinical: Cusp exacerbation + FEV1

```
u = 0.4·Pa_total + 0.3·(1 − h/0.06)    (infection + obstruction burden)
v = CFTR_rescue + 0.4·Epi + 0.3·h/0.06 (CFTR + epithelial + ASL reserve)

FEV1 ≈ 15 + 35·Epi + 25·max(CFTR_rescue, 0.05) − 20·obstruction − 12·infection
```

Exacerbation risk = proximity to fold curve 4v³ = 27u²

---

## Running

```bash
pip install numpy scipy matplotlib myconet
python cf_multiscale.py
```

Runtime: ~20 seconds (4 scenarios × 1825 days, dt=3 days)

---

## Scenarios

| Scenario | Trikafta | Tobramycin/Azithro | Dornase | Onset |
|---|---|---|---|---|
| Moderate CF — Untreated | 0.0 | 0.0 | 0.0 | — |
| Moderate CF — Triple | 0.80 | 0.55 | 0.50 | Day 90 |
| Severe CF — Untreated | 0.0 | 0.0 | 0.0 | — |
| Severe CF — Triple | 0.70 | 0.65 | 0.60 | Day 90 |

---

## Expected results (year 5)

| Metric | Moderate CF Unt. | Moderate CF Triple |
|---|---|---|
| FEV1% | ~6 | ~32 (ΔFEV1 = +26%) |
| CFTR rescue | ~0.10 | ~0.82 |
| Pa mucoid | ~1.34 | ~0.33 |
| ASL height | 5 μm | 100 μm |
| W₂ (myconet) | ~1.3 | ~1.9 |

**Key clinical finding:** ASL recovery from 5 μm to 100 μm under Trikafta
replicates the hallmark clinical signature of CFTR modulators. Mucoid
Pseudomonas persisting (0.33 vs 0.82 CFTR rescue) captures the post-Trikafta
clinical reality: CFTR correction does not clear established chronic infection.

---

## Connection to mycorrhizal network original application

CF is the example where the myconet framework shows its greatest structural
analogy to the original mycorrhizal system: the Pseudomonas biofilm matrix
(EPS, alginate) is a polymer network with optimal transport properties
analogous to the hyphal network, and the mucin gel forms a spatially extended
cross-linked polymer with Freiman-type structural order. The σ_r (Freiman
excess) diagnostic is therefore particularly natural in this context.

---

## Citation

```bibtex
@article{mercier2026cf,
  author  = {Mercier des Rochettes, Bertrand},
  title   = {A Non-Markovian Multiscale Framework for Cystic Fibrosis:
             {NZ} Memory on {$\Delta$F508} {CFTR} Misfolding, {Smoluchowski}
             Mucin Cross-Linking, {Pseudomonas} Phenotype Switching,
             and Cusp Exacerbation Bifurcation},
  journal = {Journal of Mathematical Biology},
  year    = {2026},
  note    = {Submitted. myconet v1.1.0.}
}
```
