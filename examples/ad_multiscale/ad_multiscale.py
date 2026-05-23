"""
=============================================================================
MULTISCALE MATHEMATICAL FRAMEWORK FOR ALZHEIMER'S DISEASE
Non-Markovian Aβ Nucleation | Smoluchowski Aggregation | Villani Transport
Cusp Catastrophe MCI→AD | myconet-integrated | MMSE / PET biomarkers

Author  : Bertrand Mercier des Rochettes
Institute: Quantum Proteins AI, Cergy-Pontoise, France
Target  : Journal of Mathematical Biology (JMB)
         [Competitor: Putra et al., JMB 2025 — network aggregation AD]

Four coupled scales
  L0 Molecular-quantum : Nakajima-Zwanzig non-Markovian memory on Aβ
                         nucleation lag + BACE1 processing kinetics
                         (anchored in IFP 1985 tunneling geometry work)
  L1 Aggregation       : Smoluchowski coagulation for Aβ_i oligomers
                         + tau phosphorylation cascade (prion-like seeding)
  L2 Cellular          : M1/M2 microglia + A1/A2 astrocytes + neurons
                         → myconet Villani/HWI/Talagrand T₂ diagnostics
  L3 Clinical          : Cusp catastrophe MCI→early-AD→late-AD transitions
                         + MMSE + Aβ PET SUVr + tau PET

myconet functions used (v1.0.0):
  transport.wasserstein2, relative_entropy, fisher_information
  freiman.dissipation_lower_bound, w2_lower_bound, freiman_excess

Treatment axes modelled:
  u_imm   — anti-inflammatory (minocycline / TNF-α inhibitor)
  u_abeta — anti-amyloid (lecanemab / donanemab class)
  u_tau   — anti-tau (tau aggregation inhibitor)
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings, os
warnings.filterwarnings('ignore')

# ── myconet ──────────────────────────────────────────────────────────────────
from myconet.transport import wasserstein2, relative_entropy, fisher_information
from myconet.freiman  import dissipation_lower_bound, w2_lower_bound, freiman_excess

# ── palette ───────────────────────────────────────────────────────────────────
C = dict(
    abeta  = '#E84855',   # Aβ / amyloid — red
    tau    = '#F4A261',   # tau — amber
    neuron = '#2E86AB',   # neuronal health — teal
    micro  = '#57CC99',   # microglia M2 / repair — green
    mmse   = '#9B5DE5',   # cognitive score — violet
    nz     = '#3D405B',   # NZ memory — dark slate
    bg     = '#FAFAFA',   grid = '#E2E2E2',
)
plt.rcParams.update({
    'font.family':'serif','font.size':10,'axes.labelsize':10,
    'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.facecolor':C['bg'],'figure.facecolor':'white',
    'lines.linewidth':2.0,'legend.framealpha':0.85,
    'xtick.direction':'in','ytick.direction':'in',
})

_RNG = np.random.default_rng(2026)
EPS_T = 0.30   # myconet transport scale
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDING: cellular state → 2D point cloud for myconet
#
# Inflammatory plane : (M1_micro, A1_astro)  — neuroinflammation axis
# Neuronal plane     : (N_health, My_intact)  — neuronal integrity axis
# ε = 0.30  (normalised unit square scale)
# ═══════════════════════════════════════════════════════════════════════════

_HEALTHY_REF = None   # initialised lazily

def state_to_nodes(M1, A1, N_health, My_intact, N=50, sigma=0.04):
    pts_inf = _RNG.normal([M1,  A1],       sigma, (N,2)).clip(0, 2)
    pts_neu = _RNG.normal([N_health, My_intact], sigma, (N,2)).clip(0, 1.2)
    return np.vstack([pts_inf, pts_neu])

def get_healthy_ref():
    global _HEALTHY_REF
    if _HEALTHY_REF is None:
        _HEALTHY_REF = state_to_nodes(0.04, 0.04, 0.95, 0.95, N=60)
    return _HEALTHY_REF

def myconet_diag(M1, A1, N_health, My_intact, D=0.05):
    # Clamp all inputs to finite range before embedding
    M1 = float(np.clip(M1, 0, 3))
    A1 = float(np.clip(A1, 0, 3))
    N_health  = float(np.clip(N_health,  0, 1.5))
    My_intact = float(np.clip(My_intact, 0, 1.5))
    nodes = state_to_nodes(M1, A1, N_health, My_intact)
    nodes = nodes[np.all(np.isfinite(nodes), axis=1)]  # safety filter
    if len(nodes) < 5:
        return dict(W2=0.0,H=0.0,I_fisher=0.0,sigma_r=0.0,
                    W2_lb=0.0,Psi_lb=0.0,lam_hc=0.0,T2r=1.0)
    ref   = get_healthy_ref()
    eps   = EPS_T
    W2      = wasserstein2(nodes, ref, eps)
    H       = relative_entropy(nodes, ref, eps)
    I_fish  = fisher_information(nodes, ref, eps)
    sigma_r = freiman_excess(nodes, eps)
    W2_lb   = w2_lower_bound(sigma_r, eps)
    Psi_lb  = dissipation_lower_bound(sigma_r, eps, D=D)
    W2sq    = W2**2
    lam_hc  = 2.0*H/(W2sq+1e-10) if W2sq > 1e-10 else 0.0
    T2r     = min(lam_hc*W2sq/(2.0*H+1e-10), 2.0) if H > 1e-10 else 1.0
    return dict(W2=W2, H=H, I_fisher=I_fish, sigma_r=sigma_r,
                W2_lb=W2_lb, Psi_lb=Psi_lb, lam_hc=lam_hc, T2r=T2r)


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 0 — MOLECULAR-QUANTUM
# Nakajima-Zwanzig non-Markovian memory on Aβ nucleation
#
# BACE1 cleaves APP → Aβ monomers.  The nucleation step (monomer →
# nucleus of size n*) has a lag phase that is intrinsically non-Markovian:
# the probability of forming a critical nucleus at time t depends on the
# history of monomer concentration c(s), s < t.
#
# NZ integro-differential equation (Prony form):
#   dc_APP/dt  = prod_APP − k_BACE·c_APP − δ_APP·c_APP  (APP processing)
#   dc_mon/dt  = k_BACE·c_APP − k_nuc·c_mon + Σ m_k     (monomer with memory)
#   dm_k/dt    = γ_k·c_mon − m_k/τ_k                    (NZ memory variables)
#
# Memory kernels encode:
#   k=0: positive feedback (oligomers template monomer misfolding), τ=5 d
#   k=1: negative feedback (clearance saturation),                  τ=10 d
#   k=2: insulin/metabolic modulation of BACE1 (type-3-diabetes),   τ=20 d
#
# BACE1 quantum tunneling suppression (IFP connection):
#   k_BACE(T) = A·exp(−E_a/kT) · [1 + KIE_correction·f_tunnel(geometry)]
#   f_tunnel encodes geometry-dependent tunneling (bicyclic scaffold analogy)
# ═══════════════════════════════════════════════════════════════════════════

class NZQuantumLayer:
    """
    Non-Markovian Aβ precursor processing with quantum tunneling correction
    and Prony memory variables for nucleation lag.
    """
    # Memory kernel parameters
    GAMMA = np.array([-0.03, -0.05, -0.02])   # all inhibitory NZ kernels (stable)
    TAU   = np.array([ 5.0,  10.0,  20.0])    # relaxation times (days)

    # Kinetic parameters
    PROD_APP  = 0.10   # APP basal production (a.u./day)
    K_BACE0   = 0.15   # BACE1 baseline cleavage rate
    DELTA_APP = 0.08   # APP degradation (α-secretase route)
    K_NUC0    = 0.04   # nucleation rate (monomer → critical nucleus)
    DELTA_MON = 0.12   # monomer clearance (neprilysin / IDE)

    # Quantum tunneling parameters (IFP 1985 connection)
    # KIE(H/D) ~ 1.88 measured in bicyclic scaffold;
    # tunneling factor reduces effective BACE1 rate in constrained geometry
    F_TUNNEL  = 0.15   # tunneling fraction of k_BACE
    KIE_HD    = 1.88   # kinetic isotope effect (from IFP data)

    def bace1_rate(self, insulin, age_factor):
        """
        BACE1 rate with:
        - insulin resistance upregulation (type-3-diabetes axis)
        - quantum tunneling correction (geometry-dependent)
        """
        k_classical = self.K_BACE0 * (1 + 0.4*insulin) * age_factor
        k_tunnel    = self.F_TUNNEL * k_classical / self.KIE_HD
        return k_classical + k_tunnel

    def rhs(self, t, y, insulin, age_factor, u_abeta):
        """
        y = [c_APP, c_mon, m0, m1, m2]
        insulin    : insulin resistance index [0,1]
        age_factor : age-dependent BACE1 upregulation (1 at onset, >1 later)
        u_abeta    : anti-amyloid treatment strength [0,1]
        """
        c_APP, c_mon = y[0], y[1]
        m = y[2:]

        k_bace = self.bace1_rate(insulin, age_factor)

        dc_APP = (self.PROD_APP
                  - k_bace * c_APP
                  - self.DELTA_APP * c_APP)

        # NZ nucleation: monomer production from APP cleavage,
        # clearance by neprilysin, nucleation to oligomers
        dc_mon = (k_bace * c_APP
                  - self.K_NUC0 * c_mon
                  - self.DELTA_MON * c_mon)
        # Memory contributions (non-Markovian feedback)
        for k in range(3):
            dc_mon += m[k]
        # Anti-amyloid treatment clears monomers
        dc_mon -= u_abeta * 0.40 * c_mon
        # Saturation clearance (Michaelis-Menten) — prevents blow-up
        dc_mon -= 0.08 * c_mon / (c_mon + 0.5)

        # Memory ODEs (Prony relaxation)
        dm = np.array([
            self.GAMMA[k]*c_mon - m[k]/self.TAU[k]
            for k in range(3)
        ])

        return np.concatenate([[dc_APP, dc_mon], dm])

    def y0(self, phenotype='MCI'):
        if phenotype == 'MCI':
            return np.array([0.80, 0.25, 0.0, 0.0, 0.0])
        else:   # late-AD
            return np.array([0.90, 0.55, 0.0, 0.0, 0.0])


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1 — AGGREGATION: Smoluchowski + tau phosphorylation
#
# Truncated Smoluchowski for Aβ aggregates of sizes i = 1…N_MAX:
#   dc_i/dt = (1/2) Σ_{j=1}^{i-1} k_{j,i-j}·c_j·c_{i-j}
#             − c_i · Σ_{j=1}^{N_MAX} k_{i,j}·c_j
#             − δ_i·c_i  + f_i(c_mon)
#
# Coagulation kernel: k_{i,j} = k0·(i^(1/3) + j^(1/3))²  (diffusion-limited)
# Fragmentation:      δ_i = δ0/i  (larger aggregates fragment slower)
# Source:             f_1 = nucleation flux from NZ layer (c_mon → c_1)
#
# Tau phosphorylation (prion-like seeding by Aβ oligomers):
#   dP_tau/dt = k_seed·c_olig·(1−P_tau) + k_spread·P_tau·(1−P_tau)
#               − k_clear·P_tau − u_tau·0.5·P_tau
#   c_olig = Σ_{i=2}^{6} c_i  (toxic oligomers, sizes 2–6)
#
# Neuronal loss (driven by both plaque burden and tau tangles):
#   dN/dt = −k_Ndeath·(c_plaque + w_tau·P_tau)·N
#   c_plaque = Σ_{i=7}^{N_MAX} c_i
# ═══════════════════════════════════════════════════════════════════════════

N_MAX = 12   # truncation size for Smoluchowski

class SmoluchowskiLayer:
    """
    Truncated Smoluchowski aggregation for Aβ oligomers and fibrils,
    coupled to tau phosphorylation (prion-like) and neuronal loss.
    """
    def __init__(self):
        self.k0     = 0.010   # coagulation rate constant
        self.delta0 = 0.0002   # fragmentation constant
        self.k_seed   = 0.005  # Aβ oligomer seeding of tau
        self.k_spread = 0.003  # prion-like tau self-propagation
        self.k_clear  = 0.004  # tau clearance
        self.k_Ndeath = 0.0008  # neuronal death rate
        self.w_tau    = 1.5   # tau weight in neuronal death
        self.k_nuc    = 0.008  # nucleation flux scaling (from L0)

        # Coagulation kernel matrix k_{i,j}
        sizes = np.arange(1, N_MAX+1, dtype=float)
        I, J  = np.meshgrid(sizes, sizes, indexing='ij')
        self.K_coag = self.k0 * (I**(1/3) + J**(1/3))**2
        self.delta   = self.delta0 / sizes   # fragmentation rates

    def rhs(self, t, y, c_mon, M1_micro, u_abeta, u_tau):
        """
        y = [c_1…c_N, P_tau, N_neurons]
        c_mon : Aβ monomer concentration from L0
        M1_micro : M1 microglial activation (drives tau seeding)
        """
        c = np.clip(y[:N_MAX], 0, 1e6)   # hard clip — prevents blow-up in RK4 intermediates
        P_tau = np.clip(y[N_MAX], 0, 1)
        N_neu = np.clip(y[N_MAX+1], 0, 1)

        # Vectorised Smoluchowski (no Python inner loop)
        # Loss term: dc_i -= c_i * sum_j K_{i,j} c_j
        loss_vec = self.K_coag @ c   # shape (N_MAX,)
        dc = -c * loss_vec

        # Gain term: dc_i += 0.5 * sum_{j=1}^{i-1} K_{j,i-j} c_j c_{i-j}
        for i in range(1, N_MAX):
            for j in range(i):
                dc[i] += 0.5 * self.K_coag[j, i-j-1] * c[j] * c[i-j-1]

        # Fragmentation loss/gain (vectorised)
        dc -= self.delta * c
        dc[:-1] += 0.5 * self.delta[1:] * c[1:]

        # Source: nucleation from monomers → c_1 (size-1 species)
        dc[0] += self.k_nuc * c_mon**2
        # Anti-amyloid: enhances fragmentation of all aggregates
        dc    -= u_abeta * 0.25 * c
        # Microglia M1 inhibits Aβ clearance
        dc    -= (1 - 0.3*M1_micro) * 0.02 * c

        # Toxic oligomers (sizes 2–6) and plaques (7+)
        c_olig  = c[1:6].sum()   # sizes 2–6
        c_plaq  = c[6:].sum()    # sizes 7+

        # Tau phosphorylation (prion-like seeding + basal NFT initiation)
        dP_tau = (0.0005                              # basal age-related tau initiation
                + self.k_seed  * c_olig * (1-P_tau)
                + self.k_spread* P_tau  * (1-P_tau) * (1+0.3*M1_micro)
                - self.k_clear * P_tau
                - u_tau * 0.50 * P_tau)

        # Neuronal loss
        dN_neu = -self.k_Ndeath * (c_plaq + self.w_tau*P_tau) * N_neu
        dN_neu = max(dN_neu, -N_neu)

        return np.concatenate([dc, [dP_tau, dN_neu]])

    def y0(self, phenotype='MCI'):
        if phenotype == 'MCI':
            c0 = np.zeros(N_MAX); c0[0]=0.10; c0[1]=0.05
            return np.concatenate([c0, [0.10, 0.92]])
        else:
            c0 = np.zeros(N_MAX); c0[0]=0.30; c0[1]=0.15; c0[2]=0.08
            return np.concatenate([c0, [0.40, 0.65]])

    def pet_abeta(self, c):
        """Aβ PET SUVr proxy: total fibril load (sizes 4+)."""
        return float(c[3:].sum() * 8.0 + 1.0)   # normalised to ~1.0 healthy

    def pet_tau(self, P_tau):
        """Tau PET proxy: phospho-tau burden."""
        return float(1.0 + 4.0*P_tau)


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2 — CELLULAR: microglia + astrocytes + neurons
#           + myconet Villani / HWI / Talagrand diagnostics
#
# Variables: [M1, M2, A1, A2, N_synaptic, BDNF]
#   M1/M2    : pro/anti-inflammatory microglia
#   A1/A2    : reactive/neuroprotective astrocytes
#   N_syn    : synaptic density index [0,1]
#   BDNF     : neurotrophic support
#
# Neuroinflammation driven by Aβ oligomers and tau tangles (from L1).
# M2 microglia and A2 astrocytes promote Aβ clearance.
# ═══════════════════════════════════════════════════════════════════════════

class CellularLayer:

    def rhs(self, t, y, c_olig, c_plaq, P_tau, N_neu, u_imm):
        M1,M2,A1,A2,N_syn,BDNF = y

        inflam_signal = c_olig + 0.5*c_plaq + 0.8*P_tau   # integrated burden

        # M1 microglia: activated by Aβ/tau, suppressed by Rx
        dM1 = (0.20*inflam_signal/(1+inflam_signal)
               - 0.15*M2*M1
               - 0.12*M1
               - u_imm*0.40*M1)

        # M2 microglia: neuroprotective, supports Aβ clearance
        dM2 = (0.06*(1+0.4*BDNF)
               + 0.10*M1/(1+M1)    # M1→M2 conversion with saturation
               - 0.10*M2
               - 0.15*inflam_signal*M2)

        # A1 reactive astrocytes: driven by M1, damage synapses
        dA1 = (0.18*M1/(1+M1)
               - 0.12*A1
               - u_imm*0.25*A1)

        # A2 neuroprotective astrocytes
        dA2 = (0.08*(1+0.3*BDNF)
               + 0.08*M2
               - 0.10*A2
               - 0.10*inflam_signal*A2)

        # Synaptic density: lost by A1 + tau tangles, supported by BDNF
        dN_syn = (0.05*BDNF*(1-N_syn)
                  - 0.10*A1*N_syn
                  - 0.12*P_tau*N_syn
                  - 0.05*(1-N_neu)*N_syn)
        dN_syn = np.clip(dN_syn, -N_syn, 1-N_syn)

        # BDNF: produced by active neurons, suppressed by M1/A1
        dBDNF = (0.10*N_neu*(1+0.2*A2)
                 - 0.15*M1*BDNF
                 - 0.10*A1*BDNF
                 - 0.12*BDNF)

        return [dM1, dM2, dA1, dA2,
                float(dN_syn), dBDNF]

    def y0_mci(self):  return [0.20, 0.18, 0.15, 0.20, 0.80, 0.55]
    def y0_late(self): return [0.45, 0.10, 0.40, 0.08, 0.40, 0.25]


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3 — CLINICAL: Cusp catastrophe + MMSE + biomarkers
#
# Cusp state variable x = cognitive state (-1=severe, +1=normal)
# Control u = cognitive burden (synaptic loss + tau)
# Control v = neuroprotective reserve (BDNF + N_neu)
#
# MMSE proxy: 0–30 scale mapped from (N_syn, P_tau, N_neu)
# CDR (Clinical Dementia Rating): 0–3 from cusp state
# ═══════════════════════════════════════════════════════════════════════════

class ClinicalLayer:

    def cusp_state(self, u, v):
        u = float(np.clip(u, -5, 5))
        v = float(np.clip(v,  0, 5))
        try:
            roots = np.roots([1, 0, -v, u])
            real  = roots[np.isreal(roots)].real
            return float(real.max()) if len(real) else 0.0
        except Exception:
            return 0.0

    def mmse(self, N_syn, P_tau, N_neu, BDNF):
        """MMSE score (0–30): 30 = healthy."""
        score = (12.0*N_syn + 10.0*N_neu + 5.0*BDNF
                 - 8.0*P_tau - 3.0*(1-N_neu)*(1-N_syn))
        return float(np.clip(score, 0, 30))

    def cdr(self, x):
        """CDR from cusp state: 0=normal, 0.5=MCI, 1=mild, 2=mod, 3=severe."""
        if   x >  0.6: return 0.0
        elif x >  0.2: return 0.5
        elif x > -0.2: return 1.0
        elif x > -0.6: return 2.0
        else:          return 3.0

    def cusp_surface(self, nu=70):
        u = np.linspace(-0.8, 0.8, nu)
        v = np.linspace(-0.1, 1.6, nu)
        U, V = np.meshgrid(u, v)
        X    = np.vectorize(self.cusp_state)(U, V)
        return U, V, X


# ═══════════════════════════════════════════════════════════════════════════
# MULTISCALE INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════

DIAG_STRIDE = 90   # myconet Sinkhorn every N steps

class MultiscaleAD:

    def __init__(self, pheno='MCI'):
        self.nz   = NZQuantumLayer()
        self.smol = SmoluchowskiLayer()
        self.cell = CellularLayer()
        self.clin = ClinicalLayer()
        self.pheno= pheno

    def run(self, days=1825, dt=1.0,
            u_imm=0.0, u_abeta=0.0, u_tau=0.0, onset=365):
        """
        Integrate 4-scale AD model.
        Default: 5 years (1825 days).
        Treatment onset at day 365 (diagnosis delay).
        """
        t_arr = np.arange(0, days+dt, dt)
        N     = len(t_arr)

        # Age factor: BACE1 upregulation increases 2% per year
        def age_factor(t): return 1.0 + 0.02*(t/365)

        # Insulin resistance: slow drift (type-3-diabetes axis)
        def insulin(t): return 0.10 + 0.15*np.tanh(t/1200)

        # Initial conditions
        nz_y   = self.nz.y0(self.pheno)
        smol_y = self.smol.y0(self.pheno)
        cell_y = np.array(
            self.cell.y0_mci() if self.pheno=='MCI'
            else self.cell.y0_late(), dtype=float)

        # Storage keys
        keys_basic = ('c_APP','c_mon','c_olig','c_plaq',
                      'P_tau','N_neu','M1','M2','A1','A2',
                      'N_syn','BDNF','MMSE','CDR',
                      'PET_Ab','PET_tau','cu','cv','cx')
        keys_diag  = ('W2','H','I_fisher','sigma_r',
                      'Psi_lb','lam_hc','T2r')
        S = {k: np.full(N, np.nan) for k in keys_basic+keys_diag}
        last_diag = {}

        for n, t in enumerate(t_arr):
            ui = u_imm   if t >= onset else 0.0
            ua = u_abeta if t >= onset else 0.0
            ut = u_tau   if t >= onset else 0.0

            c_APP = nz_y[0]; c_mon = nz_y[1]
            smol_c  = smol_y[:N_MAX].clip(0)
            P_tau   = np.clip(smol_y[N_MAX],   0, 1)
            N_neu   = np.clip(smol_y[N_MAX+1], 0, 1)
            M1,M2,A1,A2,N_syn,BDNF = cell_y

            c_olig = smol_c[1:6].sum()
            c_plaq = smol_c[6:].sum()

            # Store
            S['c_APP'][n]=c_APP; S['c_mon'][n]=c_mon
            S['c_olig'][n]=c_olig; S['c_plaq'][n]=c_plaq
            S['P_tau'][n]=P_tau; S['N_neu'][n]=N_neu
            S['M1'][n]=M1; S['M2'][n]=M2
            S['A1'][n]=A1; S['A2'][n]=A2
            S['N_syn'][n]=N_syn; S['BDNF'][n]=BDNF
            S['MMSE'][n]    = self.clin.mmse(N_syn,P_tau,N_neu,BDNF)
            S['PET_Ab'][n]  = self.smol.pet_abeta(smol_c)
            S['PET_tau'][n] = self.smol.pet_tau(P_tau)

            # Cusp
            u2 = 0.4*(1-N_syn) + 0.3*P_tau
            v2 = 0.6*N_neu + 0.4*BDNF
            S['cu'][n]=u2; S['cv'][n]=v2
            cx = self.clin.cusp_state(u2, v2)
            S['cx'][n]=cx
            S['CDR'][n] = self.clin.cdr(cx)

            # myconet diagnostics (every DIAG_STRIDE steps)
            if n % DIAG_STRIDE == 0:
                last_diag = myconet_diag(M1, A1, N_neu, N_syn)
            for k in keys_diag:
                S[k][n] = last_diag.get(k, np.nan)

            # Integrate L0 (NZ quantum) — RK4
            af = age_factor(t); ins = insulin(t)
            def _nz_f(yy): return np.array(self.nz.rhs(t,yy,ins,af,ua))
            k1=_nz_f(nz_y); k2=_nz_f(nz_y+dt/2*k1)
            k3=_nz_f(nz_y+dt/2*k2); k4=_nz_f(nz_y+dt*k3)
            nz_y=(nz_y+dt/6*(k1+2*k2+2*k3+k4)).clip(0)

            # Integrate L1 (Smoluchowski + tau) — RK4 with clipping
            def _sm_f(yy):
                yc = yy.copy(); yc[:N_MAX]=yc[:N_MAX].clip(0)
                return np.array(self.smol.rhs(t,yc,c_mon,M1,ua,ut))
            k1=_sm_f(smol_y); k2=_sm_f(smol_y+dt/2*k1)
            k3=_sm_f(smol_y+dt/2*k2); k4=_sm_f(smol_y+dt*k3)
            smol_y=smol_y+dt/6*(k1+2*k2+2*k3+k4)
            smol_y[:N_MAX]=smol_y[:N_MAX].clip(0)
            smol_y[N_MAX]=np.clip(smol_y[N_MAX],0,1)
            smol_y[N_MAX+1]=np.clip(smol_y[N_MAX+1],0,1)

            # Integrate L2 (cellular) — RK4
            def _ce_f(yy): return np.array(self.cell.rhs(t,yy,c_olig,c_plaq,P_tau,N_neu,ui))
            k1=_ce_f(cell_y); k2=_ce_f(cell_y+dt/2*k1)
            k3=_ce_f(cell_y+dt/2*k2); k4=_ce_f(cell_y+dt*k3)
            cell_y=(cell_y+dt/6*(k1+2*k2+2*k3+k4)).clip(0)
            cell_y[4]=np.clip(cell_y[4],0,1)

        S['t'] = t_arr
        return S


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

def run_scenarios():
    cfgs = [
        ('MCI',  0.0,  0.0,  0.0,  'MCI — Untreated'),
        ('MCI',  0.50, 0.60, 0.40, 'MCI — Triple Therapy'),
        ('late', 0.0,  0.0,  0.0,  'Late-AD — Untreated'),
        ('late', 0.40, 0.70, 0.55, 'Late-AD — Triple Therapy'),
    ]
    S = {}
    for pheno,ui,ua,ut,label in cfgs:
        print(f'  {label} …', end=' ', flush=True)
        m = MultiscaleAD(pheno)
        S[label] = m.run(days=1825, dt=1.0,
                         u_imm=ui, u_abeta=ua, u_tau=ut, onset=365)
        print('done')
    return S


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def fig1_clinical(S):
    fig,axes = plt.subplots(3,2,figsize=(13,10))
    fig.suptitle('Figure 1 — Clinical Trajectories: MCI vs Late-AD\n'
                 'MMSE score, Aβ PET SUVr, Tau PET (untreated vs triple therapy)',
                 fontweight='bold', fontsize=12)
    rows = [('MMSE','MMSE (0–30)'),('PET_Ab','Aβ PET SUVr'),('PET_tau','Tau PET')]
    phenos = [('MCI','MCI → early AD'),('late','Late-AD')]
    for ci,(ph,coltitle) in enumerate(phenos):
        keys_p = [k for k in S if ph in k]
        for ri,(var,ylab) in enumerate(rows):
            ax = axes[ri,ci]
            for key in keys_p:
                s = S[key]
                clr = C['abeta'] if 'Untreated' in key else C['neuron']
                ls  = '-' if 'Untreated' in key else '--'
                lb  = 'Untreated' if 'Untreated' in key else 'Triple Therapy'
                ax.plot(s['t']/365, s[var], color=clr, ls=ls,
                        lw=2.0, label=lb, alpha=0.92)
            ax.axvline(1, color='gray', ls=':', lw=1.0, alpha=0.6)
            if ri==0: ax.set_title(coltitle, fontweight='bold')
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_xlabel('Time (years)', fontsize=9)
            ax.set_xlim(0,5)
            if var=='MMSE':
                ax.set_ylim(0,32)
                for lv,lt in [(24,'Mild dementia'),(18,'Mod. dementia')]:
                    ax.axhline(lv,color=C['mmse'],ls=':',lw=0.8,alpha=0.5)
                    ax.text(4.0,lv+0.3,lt,fontsize=7,color=C['mmse'])
            ax.grid(True,color=C['grid'],lw=0.4)
            ax.legend(fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(f'{OUTDIR}/fig1_clinical.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ Fig 1')


def fig2_smoluchowski(S):
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    fig.suptitle('Figure 2 — L0+L1: NZ Quantum Nucleation & Smoluchowski Aggregation\n'
                 'APP processing (BACE1+tunneling), Aβ monomer, oligomers, plaques, '
                 'tau phosphorylation, neuronal loss',
                 fontweight='bold', fontsize=12)
    info = [
        ('c_APP','APP concentration','myconet — NZ L0',C['nz']),
        ('c_mon','Aβ monomer c₁ (NZ nucleation)','L0→L1 interface',C['abeta']),
        ('c_olig','Aβ oligomers (sizes 2–6)','L1 Smoluchowski',C['abeta']),
        ('c_plaq','Aβ plaques (sizes 7–12)','L1 Smoluchowski',C['tau']),
        ('P_tau','Tau phosphorylation P_τ','L1 prion-like seeding',C['tau']),
        ('N_neu','Neuronal survival N','L1→L2 interface',C['neuron']),
    ]
    for ax,(var,title,src,clr) in zip(axes.flat,info):
        for key,s in S.items():
            ls = '-' if 'Untreated' in key else '--'
            al = 0.92 if 'MCI' in key else 0.55
            c2 = clr if 'MCI' in key else '#999999'
            lb = ('MCI ' if 'MCI' in key else 'Late ') + \
                 ('Unt.' if 'Untreated' in key else 'Trt.')
            ax.plot(s['t']/365, s[var], color=c2, ls=ls,
                    lw=1.9, alpha=al, label=lb)
        ax.axvline(1, color='gray', ls=':', lw=0.9, alpha=0.6)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.set_xlabel('Time (years)', fontsize=8)
        ax.text(0.99,0.02,src,transform=ax.transAxes,
                fontsize=6,ha='right',color='#888',style='italic')
        ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(f'{OUTDIR}/fig2_smoluchowski.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ Fig 2')


def fig3_villani(S):
    fig,axes = plt.subplots(2,3,figsize=(15,9))
    fig.suptitle('Figure 3 — L2 Cellular: myconet Villani/HWI/Talagrand Diagnostics\n'
                 'W₂ (Sinkhorn), H (KL), I (Fisher), Ψ_lb (dissipation), '
                 'σ_r (Freiman), ρ_T (T₂ saturation)',
                 fontweight='bold', fontsize=12)
    diag_info = [
        ('W2',     'W₂ distance (Sinkhorn)',       'myconet.transport.wasserstein2'),
        ('H',      'Relative entropy H(ρ|ρ∞)',      'myconet.transport.relative_entropy'),
        ('I_fisher','Fisher information I(ρ|ρ∞)',   'myconet.transport.fisher_information'),
        ('Psi_lb', 'Villani dissipation Ψ_lb',      'myconet.freiman.dissipation_lower_bound'),
        ('sigma_r','Freiman excess σ_r − K_HEX',    'myconet.freiman.freiman_excess'),
        ('T2r',    'Talagrand T₂ saturation ρ_T',   'myconet.freiman.w2_lower_bound'),
    ]
    for ax,(var,title,src) in zip(axes.flat,diag_info):
        for key,s in S.items():
            clr = C['neuron'] if 'MCI' in key else C['abeta']
            ls  = '-' if 'Untreated' in key else '--'
            lb  = ('MCI ' if 'MCI' in key else 'Late ') + \
                  ('Unt.' if 'Untreated' in key else 'Trt.')
            vals = np.clip(s[var],0,1.5) if var=='T2r' else s[var]
            ax.plot(s['t']/365, vals, color=clr, ls=ls,
                    lw=1.8, alpha=0.85, label=lb)
        if var=='T2r':
            ax.axhline(1.0,color='k',ls=':',lw=1.0,label='T₂ saturation')
        ax.set_title(title,fontweight='bold',fontsize=9)
        ax.set_xlabel('Time (years)',fontsize=8)
        ax.text(0.99,0.02,src,transform=ax.transAxes,
                fontsize=6,ha='right',color='#888',style='italic')
        ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(f'{OUTDIR}/fig3_villani.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ Fig 3')


def fig4_cusp(S):
    fig = plt.figure(figsize=(15,6))
    fig.suptitle('Figure 4 — L3 Clinical: Cusp Catastrophe & MCI→AD Bifurcation\n'
                 'V(x;u,v)=x⁴/4−v·x²/2+u·x   |   u: cognitive burden   '
                 'v: neuroprotective reserve',
                 fontweight='bold',fontsize=12)
    clin = ClinicalLayer()

    ax3d = fig.add_subplot(131,projection='3d')
    U,V,X = clin.cusp_surface(nu=65)
    ax3d.plot_surface(U,V,X,cmap='RdYlBu_r',alpha=0.72,rstride=2,cstride=2)
    ax3d.set_xlabel('u (burden)',fontsize=7,labelpad=1)
    ax3d.set_ylabel('v (reserve)',fontsize=7,labelpad=1)
    ax3d.set_zlabel('x (cognition)',fontsize=7,labelpad=1)
    ax3d.set_title('Cusp surface',fontweight='bold',fontsize=10)
    ax3d.view_init(elev=22,azim=-58)
    # Annotate the two clinical zones
    ax3d.text2D(0.05,0.90,'Cognitively\nnormal',transform=ax3d.transAxes,
                fontsize=7,color=C['neuron'])
    ax3d.text2D(0.65,0.10,'Dementia',transform=ax3d.transAxes,
                fontsize=7,color=C['abeta'])

    ax2 = fig.add_subplot(132)
    uf = np.linspace(0.01,0.8,300)
    vf = (27*uf**2/4)**(1/3)
    ax2.fill_between(uf,vf,1.65,color=C['neuron'],alpha=0.12)
    ax2.fill_between(-uf,vf,1.65,color=C['neuron'],alpha=0.12)
    ax2.plot(uf,vf,color=C['nz'],lw=2.0)
    ax2.plot(-uf,vf,color=C['nz'],lw=2.0)
    ax2.plot(0,0,'k*',ms=12,label='Cusp point')
    ax2.text(0.05,0.25,'Bistable\nMCI zone',fontsize=8,color=C['nz'])
    for key,s in S.items():
        clr = C['neuron'] if 'MCI' in key else C['abeta']
        ls  = '-' if 'Untreated' in key else '--'
        lb  = ('MCI ' if 'MCI' in key else 'Late ') + \
              ('Unt.' if 'Untreated' in key else 'Trt.')
        ax2.plot(s['cu'],s['cv'],color=clr,ls=ls,lw=1.3,alpha=0.75,label=lb)
        ax2.plot(s['cu'][0],s['cv'][0],'o',color=clr,ms=5)
        ax2.plot(s['cu'][-1],s['cv'][-1],'s',color=clr,ms=5)
    ax2.set_xlim(-0.05,0.95); ax2.set_ylim(-0.05,1.65)
    ax2.set_xlabel('u — Cognitive burden')
    ax2.set_ylabel('v — Neuroprotective reserve')
    ax2.set_title('Control-space trajectories',fontweight='bold',fontsize=10)
    ax2.legend(fontsize=7,ncol=2); ax2.grid(True,color=C['grid'],lw=0.4)

    axb = fig.add_subplot(133)
    for key,s in S.items():
        clr = C['neuron'] if 'MCI' in key else C['abeta']
        ls  = '-' if 'Untreated' in key else '--'
        lb  = ('MCI ' if 'MCI' in key else 'Late ') + \
              ('Unt.' if 'Untreated' in key else 'Trt.')
        axb.plot(s['t']/365,s['cx'],color=clr,ls=ls,lw=1.8,alpha=0.85,label=lb)
    axb.axvline(1,color='gray',ls=':',lw=1.0,alpha=0.7)
    axb.axhline(0.2,color=C['mmse'],ls=':',lw=0.8,alpha=0.6)
    axb.text(4.0,0.22,'MCI threshold',fontsize=7,color=C['mmse'])
    axb.set_xlabel('Time (years)')
    axb.set_ylabel('x — Cognitive state')
    axb.set_title('MCI→AD transition x(t)',fontweight='bold',fontsize=10)
    axb.legend(fontsize=7); axb.grid(True,color=C['grid'],lw=0.4)

    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig(f'{OUTDIR}/fig4_cusp.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ Fig 4')


def fig5_cellular(S):
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    fig.suptitle('Figure 5 — L2 Cellular: Neuroinflammation & Neuroprotection\n'
                 'M1/M2 microglia, A1/A2 astrocytes, synaptic density, BDNF',
                 fontweight='bold',fontsize=12)
    info=[('M1','M1 microglia (pro-inflam.)',C['abeta']),
          ('M2','M2 microglia (neuroprot.)',C['micro']),
          ('A1','A1 astrocytes (reactive)',C['tau']),
          ('A2','A2 astrocytes (neuroprot.)',C['neuron']),
          ('N_syn','Synaptic density',C['mmse']),
          ('BDNF','BDNF neurotrophic factor',C['nz'])]
    for ax,(var,title,clr) in zip(axes.flat,info):
        for key,s in S.items():
            ls = '-' if 'Untreated' in key else '--'
            al = 0.92 if 'MCI' in key else 0.55
            c2 = clr if 'MCI' in key else '#999999'
            lb = ('MCI ' if 'MCI' in key else 'Late ') + \
                 ('Unt.' if 'Untreated' in key else 'Trt.')
            ax.plot(s['t']/365,s[var],color=c2,ls=ls,lw=1.9,alpha=al,label=lb)
        ax.axvline(1,color='gray',ls=':',lw=0.9,alpha=0.6)
        ax.set_title(title,fontweight='bold',fontsize=9)
        ax.set_xlabel('Time (years)',fontsize=8)
        ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(f'{OUTDIR}/fig5_cellular.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ Fig 5')


def fig6_summary(S):
    fig = plt.figure(figsize=(16,11))
    fig.suptitle(
        'Figure 6 — 4-Scale Integration: MCI Untreated vs Triple Therapy\n'
        'L0 NZ Quantum  ·  L1 Smoluchowski  ·  L2 myconet Villani  ·  L3 Cusp Clinical',
        fontweight='bold',fontsize=12)
    gs = gridspec.GridSpec(4,3,figure=fig,hspace=0.56,wspace=0.42)

    su = S['MCI — Untreated']
    st = S['MCI — Triple Therapy']
    t  = su['t']/365

    def panel(pos,var,ylab,title,ylim=None):
        ax = fig.add_subplot(pos)
        ax.plot(t,su[var],color=C['abeta'],lw=2.0,label='Untreated')
        ax.plot(t,st[var],color=C['neuron'],lw=2.2,ls='--',label='Triple Therapy')
        ax.axvline(1,color='gray',ls=':',lw=0.9,alpha=0.6)
        ax.set_title(title,fontsize=8.5,fontweight='bold')
        ax.set_ylabel(ylab,fontsize=8); ax.set_xlabel('Years',fontsize=8)
        ax.grid(True,color=C['grid'],lw=0.35)
        if ylim: ax.set_ylim(ylim)
        return ax

    # L0 — NZ Quantum
    panel(gs[0,0],'c_APP','APP','L0: APP (BACE1+NZ)')
    panel(gs[0,1],'c_mon','Aβ₁','L0→L1: Aβ monomer (NZ memory)')
    panel(gs[0,2],'c_olig','Aβ oligo','L1: Aβ oligomers (Smoluchowski)')

    # L1 — Smoluchowski
    panel(gs[1,0],'c_plaq','Aβ plaque','L1: Aβ plaques')
    panel(gs[1,1],'P_tau','P_τ','L1: Tau phosphorylation',(0,1))
    panel(gs[1,2],'N_neu','N','L1→L2: Neuronal survival',(0,1))

    # L2 — Cellular / myconet
    panel(gs[2,0],'M1','M1','L2: M1 microglia')
    panel(gs[2,1],'W2','W₂','L2: myconet W₂ (Sinkhorn)')
    panel(gs[2,2],'T2r','ρ_T','L2: myconet T₂ ratio',(0,1.4))

    # L3 — Clinical
    panel(gs[3,0],'MMSE','MMSE','L3: MMSE score',(0,32))
    panel(gs[3,1],'PET_Ab','SUVr','L3: Aβ PET SUVr')
    panel(gs[3,2],'PET_tau','tSUVr','L3: Tau PET')

    # Scale labels
    for row,label,clr in [
            (0,'L0 QUANTUM\n(NZ+BACE1)',C['nz']),
            (1,'L1 AGGREG.\n(Smoluch.)',C['abeta']),
            (2,'L2 CELLULAR\n(myconet)',C['micro']),
            (3,'L3 CLINICAL\n(Cusp)',C['mmse'])]:
        fig.text(0.003,0.87-row*0.225,label,fontsize=7,fontweight='bold',
                 color=clr,va='center',rotation=90,transform=fig.transFigure)

    from matplotlib.lines import Line2D
    hdl=[Line2D([0],[0],color=C['abeta'],lw=2.0,label='Untreated'),
         Line2D([0],[0],color=C['neuron'],lw=2.2,ls='--',label='Triple Therapy'),
         Line2D([0],[0],color='gray',lw=1.0,ls=':',label='Treatment onset (yr 1)')]
    fig.legend(handles=hdl,loc='lower center',ncol=3,fontsize=9,
               bbox_to_anchor=(0.5,0.0),framealpha=0.9)

    plt.savefig(f'{OUTDIR}/fig6_summary.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ Fig 6')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n'+'═'*64)
    print('  MULTISCALE AD MODEL — myconet-integrated edition')
    print('  Mercier des Rochettes (2026)  |  Quantum Proteins AI')
    print('  Target: Journal of Mathematical Biology')
    print('  Competitor: Putra et al., JMB 2025 (network aggregation)')
    print('═'*64+'\n')

    print('► Simulations (4 scenarios × 1825 days) …')
    S = run_scenarios()

    print('\n► Paper-quality figures …')
    fig1_clinical(S)
    fig2_smoluchowski(S)
    fig3_villani(S)
    fig4_cusp(S)
    fig5_cellular(S)
    fig6_summary(S)
    fig7_dual_w2(S)

    print('\n'+'═'*64)
    for ph in ('MCI','late'):
        tag = 'MCI' if ph=='MCI' else 'Late-AD'
        su = S[f'{tag} — Untreated']
        st = S[f'{tag} — Triple Therapy']
        print(f'  {tag}  MMSE    yr5 : Unt={su["MMSE"][-1]:.1f}'
              f'  Trt={st["MMSE"][-1]:.1f}'
              f'  Δ={st["MMSE"][-1]-su["MMSE"][-1]:+.1f}')
        print(f'  {tag}  P_tau   yr5 : Unt={su["P_tau"][-1]:.3f}'
              f'  Trt={st["P_tau"][-1]:.3f}')
        print(f'  {tag}  N_neu   yr5 : Unt={su["N_neu"][-1]:.3f}'
              f'  Trt={st["N_neu"][-1]:.3f}')
        print(f'  {tag}  PET_Aβ yr5 : Unt={su["PET_Ab"][-1]:.2f}'
              f'  Trt={st["PET_Ab"][-1]:.2f}')
        print(f'  {tag}  W₂     yr5 : Unt={su["W2"][-1]:.3f}'
              f'  Trt={st["W2"][-1]:.3f}')
        print(f'  {tag}  σ_r    yr5 : Unt={su["sigma_r"][-1]:.3f}'
              f'  Trt={st["sigma_r"][-1]:.3f}')
        print()
    print(f'  Figures → {OUTDIR}/')
    print('═'*64+'\n')
