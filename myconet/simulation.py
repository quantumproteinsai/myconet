"""
myconet.simulation
==================
Main MycoNetSimulation class integrating:
  - Stochastic hyphal growth (HyphalNetwork)
  - Fokker–Planck nutrient transport
  - Freiman index tracking (freiman.py)
  - Wasserstein distance and Fisher information (transport.py)

Reproduces Figure 1 and Table 1 of:
  Mercier des Rochettes, B. (2026). "Geometric Efficiency Bounds for
  Mycorrhizal Networks: A Freiman–Villani Framework."
  Journal of Mathematical Biology (preprint).

Usage
-----
>>> from myconet.simulation import MycoNetSimulation
>>> sim = MycoNetSimulation(seed=42)
>>> results = sim.run(T=120, drought_onset=48)
>>> results.plot()
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List

from .network import HyphalNetwork, hexagonal_lattice, network_drift
from .freiman import (
    local_freiman_index, K_HEX, c0, C_STAR, dissipation_lower_bound
)
from .transport import (
    fokker_planck_step, growth_signal_step, wasserstein2, fisher_information
)


@dataclass
class SimulationParams:
    """
    Biophysical parameters for the mycorrhizal simulation.
    All dimensional quantities in CGS units (cm, h).
    """
    # Domain
    domain_size : float = 10.0      # [cm] side length of square domain
    grid_res    : int   = 50        # grid points per dimension for PDE
    eps         : float = 0.01      # [cm] mean hyphal spacing

    # Diffusion / transport
    D           : float = 3.6e-3   # [cm^2/h] phosphorus diffusion (=1e-6 cm^2/s)
    D_c         : float = 1.8e-3   # [cm^2/h] growth signal diffusion (=5e-7 cm^2/s)

    # Growth signal kinetics
    alpha       : float = 0.8      # [1/h] coupling rho -> c
    beta        : float = 0.3      # [1/h] decay rate of c

    # Branching
    alpha_b     : float = 0.5      # [1/h] branching rate coefficient
    sigma_noise : float = 0.30     # directional noise in branching
    max_nodes   : int   = 2000     # upper limit on node count

    # Time stepping
    dt          : float = 0.2      # [h] time step
    record_every: int   = 5        # record diagnostics every N steps

    # Freiman index
    k_neighbors : int   = 7        # kNN local set size (k=7 gives K_HEX=19/7)

    # Initial conditions
    N_init      : int   = 200      # initial number of nodes


@dataclass
class SimulationResults:
    """Container for simulation output time series."""
    time       : np.ndarray
    sigma_r    : np.ndarray   # local Freiman index
    w2_eps     : np.ndarray   # W2(mu_Gamma, mu_hex) / eps
    fisher     : np.ndarray   # Fisher information I(mu_Gamma | mu_hex) [cm^-2]
    psi        : np.ndarray   # metabolic dissipation Psi = D * I  [1/h]
    psi_bound  : np.ndarray   # Theorem 6.1 lower bound on Psi
    n_nodes    : np.ndarray   # number of nodes at each recorded step
    params     : SimulationParams = field(default_factory=SimulationParams)

    def phase(self):
        """Classify each time point into Healthy / Adaptive / Stressed."""
        phases = np.empty(len(self.sigma_r), dtype=object)
        phases[self.sigma_r <= 3.2]                          = 'Healthy'
        phases[(self.sigma_r > 3.2) & (self.sigma_r <= 4.5)] = 'Adaptive'
        phases[self.sigma_r > 4.5]                           = 'Stressed'
        return phases

    def psi_ratio(self):
        """Psi(t) / Psi(t=0): metabolic dissipation normalized to initial value."""
        if self.psi[0] > 0:
            return self.psi / self.psi[0]
        return self.psi

    def summary(self):
        drought_idx = np.searchsorted(self.time, 48)
        print(f"{'Quantity':<35} {'Pre-drought':>14} {'Post-drought':>14}")
        print("-" * 65)
        for name, arr in [
            ("sigma_r (Freiman index)", self.sigma_r),
            ("W2/eps (Wasserstein)", self.w2_eps),
            ("Psi [1/h] (dissipation)", self.psi),
            ("Psi / Psi_hex (ratio)", self.psi_ratio()),
        ]:
            pre  = arr[:drought_idx].mean() if drought_idx > 0 else arr[0]
            post = arr[drought_idx:].mean()
            print(f"{name:<35} {pre:>14.4f} {post:>14.4f}")
        print()
        ex_pre  = max(0, self.sigma_r[:drought_idx].mean() - K_HEX)
        ex_post = max(0, self.sigma_r[drought_idx:].mean() - K_HEX)
        if ex_pre > 0:
            ratio = (ex_post / ex_pre)**2
            print(f"Predicted Psi ratio (sigma_r scaling)^2 = {ratio:.2f}")

    def plot(self, save_path=None):
        """Plot the three-panel Figure 1 of the paper."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(9, 8))
        gs = gridspec.GridSpec(3, 1, hspace=0.38)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        drought = 48.0
        colors = {'sigma': '#2a5c8a', 'w2': '#2a6a3a', 'psi': '#8a2a2a'}

        # ── Panel (a): Freiman index ──
        ax1.axvline(drought, color='grey', lw=1.2, ls='--', alpha=0.7)
        ax1.axhline(K_HEX, color='grey', lw=0.9, ls=':', alpha=0.7)
        ax1.plot(self.time, self.sigma_r, color=colors['sigma'], lw=1.8)
        ax1.text(drought + 0.5, self.sigma_r.max() * 0.97,
                 'drought onset', fontsize=8, color='grey')
        ax1.text(self.time[-1] * 0.02, K_HEX + 0.03,
                 r'$K_{\rm hex}=19/7$', fontsize=8, color='grey')
        ax1.set_ylabel(r'$\sigma_r(\Gamma)$', fontsize=11)
        ax1.set_title('(a) Local Freiman Index', fontsize=10, loc='left')
        ax1.set_xlim(self.time[0], self.time[-1])

        # ── Panel (b): W2/eps ──
        ax2.axvline(drought, color='grey', lw=1.2, ls='--', alpha=0.7)
        ax2.plot(self.time, self.w2_eps, color=colors['w2'], lw=1.8)
        # Proposition 4.4 lower bound
        w2_bound = c0 * np.maximum(self.sigma_r - K_HEX, 0)
        ax2.plot(self.time, w2_bound, color=colors['w2'], lw=1.0,
                 ls='--', alpha=0.6, label='Prop. 4.4 lower bound')
        ax2.legend(fontsize=8, framealpha=0.7)
        ax2.set_ylabel(r'$W_2(\mu_\Gamma, \mu_{\rm hex})/\varepsilon$', fontsize=11)
        ax2.set_title('(b) Wasserstein Distance', fontsize=10, loc='left')
        ax2.set_xlim(self.time[0], self.time[-1])

        # ── Panel (c): Psi ──
        ax3.axvline(drought, color='grey', lw=1.2, ls='--', alpha=0.7)
        ax3.plot(self.time, self.psi, color=colors['psi'], lw=1.8, label='Simulation')
        ax3.plot(self.time, self.psi_bound, color=colors['psi'], lw=1.0,
                 ls='--', alpha=0.6, label='Thm. 6.1 lower bound')
        ax3.legend(fontsize=8, framealpha=0.7)
        ax3.set_ylabel(r'$\Psi\,\varepsilon^2/D$', fontsize=11)
        ax3.set_xlabel('Time (h)', fontsize=11)
        ax3.set_title('(c) Metabolic Dissipation', fontsize=10, loc='left')
        ax3.set_xlim(self.time[0], self.time[-1])

        # ── Inset: Psi vs (sigma_r - K_hex)^2 ──
        ax_ins = ax3.inset_axes([0.62, 0.08, 0.35, 0.55])
        ex2 = np.maximum(self.sigma_r - K_HEX, 0)**2
        psi_norm = self.psi / (self.params.D / self.params.eps**2)
        ax_ins.scatter(ex2, psi_norm, s=3, alpha=0.4, color=colors['psi'])
        # Regression line
        valid = ex2 > 1e-4
        if valid.sum() > 5:
            slope = np.polyfit(ex2[valid], psi_norm[valid], 1)[0]
            x_fit = np.linspace(0, ex2.max(), 50)
            ax_ins.plot(x_fit, slope * x_fit, 'r-', lw=1.2)
            ax_ins.text(0.05, 0.88, f'$C_{{sim}}≈{slope:.1f}$',
                       transform=ax_ins.transAxes, fontsize=7.5, color='red')
        ax_ins.set_xlabel(r'$(\sigma_r - K_{{\rm hex}})^2$', fontsize=7)
        ax_ins.set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=7)
        ax_ins.tick_params(labelsize=6.5)

        plt.suptitle(
            'MycoNet Simulation: Freiman–Villani Analysis under Drought Stress',
            fontsize=11, y=0.995
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()


class MycoNetSimulation:
    """
    Full mycorrhizal network simulation with Freiman–Villani analysis.

    Parameters
    ----------
    params : SimulationParams or None (uses defaults)
    seed   : int, random seed for reproducibility
    """

    def __init__(self, params=None, seed=42):
        self.params = params or SimulationParams()
        self.rng = np.random.default_rng(seed)
        p = self.params

        # PDE grid
        self.grid_x = np.linspace(0, p.domain_size, p.grid_res)
        self.grid_y = np.linspace(0, p.domain_size, p.grid_res)
        self.hx = self.grid_x[1] - self.grid_x[0]
        self.hy = self.grid_y[1] - self.grid_y[0]
        XX, YY = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
        self.XX, self.YY = XX, YY

        # Hexagonal reference lattice (fixed throughout simulation)
        self.hex_nodes = hexagonal_lattice(p.eps, p.domain_size)

        # CFL stability check
        cfl = p.D * p.dt / min(self.hx, self.hy)**2
        if cfl > 0.4:
            import warnings
            warnings.warn(
                f"CFL number {cfl:.3f} > 0.4 — consider reducing dt or increasing grid_res.",
                RuntimeWarning
            )

    def _make_source(self, t, drought_onset, drought_strength):
        """Nutrient source term: root tips in bottom-left, sink on boundary."""
        p = self.params
        source = np.zeros((p.grid_res, p.grid_res))
        # Source: gaussian cluster near (1,1) cm
        r2 = (self.XX - 1.0)**2 + (self.YY - 1.0)**2
        source += 0.5 * np.exp(-r2 / (2 * (0.5)**2))
        # Drought: reduce source strength after onset
        if t > drought_onset:
            strength = 1.0 - drought_strength * min((t - drought_onset) / 20.0, 1.0)
            source *= max(strength, 0.05)
        return source

    def run(self, T=120.0, drought_onset=48.0, drought_strength=0.85,
            verbose=True):
        """
        Run the simulation from t=0 to t=T [h] with drought stress at
        t=drought_onset.

        Parameters
        ----------
        T               : float, total simulation time [h]
        drought_onset   : float, time at which drought starts [h]
        drought_strength: float in [0,1], severity (0=none, 1=complete)
        verbose         : bool, print progress

        Returns
        -------
        SimulationResults
        """
        p = self.params
        n_steps = int(T / p.dt)
        record_idx = list(range(0, n_steps + 1, p.record_every))

        # ── Initialise ──
        net = HyphalNetwork.random_seed(
            p.N_init, domain_size=p.domain_size, eps=p.eps, rng=self.rng
        )

        # Initial PDE fields: rho ~ uniform, c ~ 0
        rho = np.ones((p.grid_res, p.grid_res)) / (p.domain_size**2)
        c   = np.zeros((p.grid_res, p.grid_res))

        # Storage
        times, sigma_r_list, w2_list = [], [], []
        fisher_list, psi_list, psi_bound_list, nn_list = [], [], [], []

        t_start = time.time()

        for step in range(n_steps + 1):
            t = step * p.dt

            # ── Record diagnostics ──
            if step in record_idx:
                # Freiman index
                sig, _ = local_freiman_index(net.nodes, p.eps, k=7)

                # Wasserstein distance
                w2 = wasserstein2(net.nodes, self.hex_nodes, p.eps)

                # Fisher information → dissipation
                FI = fisher_information(net.nodes, self.hex_nodes, p.eps,
                                        grid_size=30)
                psi = p.D * FI

                # Theorem 6.1 lower bound
                pb = dissipation_lower_bound(sig, p.eps, p.D)

                times.append(t)
                sigma_r_list.append(sig)
                w2_list.append(w2)
                fisher_list.append(FI)
                psi_list.append(psi / (p.D / p.eps**2))   # normalized
                psi_bound_list.append(pb / (p.D / p.eps**2))
                nn_list.append(net.N)

                if verbose and step % (10 * p.record_every) == 0:
                    elapsed = time.time() - t_start
                    print(f"  t={t:6.1f}h  σ_r={sig:.3f}  W₂/ε={w2:.3f}"
                          f"  Ψε²/D={psi_list[-1]:.2f}  N={net.N}"
                          f"  [{elapsed:.0f}s elapsed]")

            if step == n_steps:
                break

            # ── Update PDE fields ──
            source = self._make_source(t, drought_onset, drought_strength)

            # Drift from current network geometry
            vx, vy = network_drift(
                net.nodes, p.eps, self.grid_x, self.grid_y, p.D
            )
            # Clip drift to maintain numerical stability
            v_max = 0.5 * min(self.hx, self.hy) / p.dt
            vx = np.clip(vx, -v_max, v_max)
            vy = np.clip(vy, -v_max, v_max)

            rho = fokker_planck_step(rho, vx, vy, source, p.D, p.dt, self.hx, self.hy)
            c   = growth_signal_step(c, rho, p.alpha, p.beta, p.D_c, p.dt,
                                     self.hx, self.hy)

            # ── Update network ──
            net.branch(c, self.grid_x, self.grid_y, p.dt,
                       alpha_b=p.alpha_b, sigma_noise=p.sigma_noise,
                       max_nodes=p.max_nodes)

            # Prune under drought
            if t > drought_onset:
                net.prune(c, self.grid_x, self.grid_y, p.dt,
                          beta_p=0.05 * drought_strength)

        if verbose:
            print(f"\nSimulation complete ({time.time()-t_start:.1f}s)")

        results = SimulationResults(
            time       = np.array(times),
            sigma_r    = np.array(sigma_r_list),
            w2_eps     = np.array(w2_list),
            fisher     = np.array(fisher_list),
            psi        = np.array(psi_list),
            psi_bound  = np.array(psi_bound_list),
            n_nodes    = np.array(nn_list),
            params     = p,
        )
        return results


def run_ensemble(n_runs=10, T=120.0, drought_onset=48.0,
                 drought_strength=0.85, verbose=False, **kwargs):
    """
    Run n_runs independent simulations and return mean ± std time series.

    Returns
    -------
    dict with keys: time, sigma_r_mean, sigma_r_std, w2_mean, w2_std,
                    psi_mean, psi_std, psi_bound_mean
    """
    all_results = []
    for i in range(n_runs):
        if verbose:
            print(f"\n── Run {i+1}/{n_runs} ──")
        sim = MycoNetSimulation(seed=i * 137 + 1, **kwargs)
        r = sim.run(T=T, drought_onset=drought_onset,
                    drought_strength=drought_strength, verbose=verbose)
        all_results.append(r)

    # Interpolate to common time grid
    t_ref = all_results[0].time
    def stack(attr):
        return np.array([getattr(r, attr) for r in all_results])

    sig_arr = stack('sigma_r')
    w2_arr  = stack('w2_eps')
    psi_arr = stack('psi')
    pb_arr  = stack('psi_bound')

    return {
        'time'          : t_ref,
        'sigma_r_mean'  : sig_arr.mean(axis=0),
        'sigma_r_std'   : sig_arr.std(axis=0),
        'w2_mean'       : w2_arr.mean(axis=0),
        'w2_std'        : w2_arr.std(axis=0),
        'psi_mean'      : psi_arr.mean(axis=0),
        'psi_std'       : psi_arr.std(axis=0),
        'psi_bound_mean': pb_arr.mean(axis=0),
        'all_results'   : all_results,
    }
