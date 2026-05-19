"""
examples/drought_stress.py
==========================
Reproduces Figure 1 and Table 1 of:

  Mercier des Rochettes, B. (2026).
  "Geometric Efficiency Bounds for Mycorrhizal Networks:
   A Freiman–Villani Framework."
  Journal of Mathematical Biology.

Usage
-----
    python examples/drought_stress.py                # single run (fast)
    python examples/drought_stress.py --ensemble     # 10 runs (paper quality)
    python examples/drought_stress.py --save fig1.png
"""

import sys
import os
import argparse
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from myconet import (MycoNetSimulation, SimulationParams,
                     run_ensemble, K_HEX)


def parse_args():
    p = argparse.ArgumentParser(description='Drought stress experiment')
    p.add_argument('--ensemble', action='store_true',
                   help='Run 10-run ensemble (slower, matches paper)')
    p.add_argument('--n-runs', type=int, default=10)
    p.add_argument('--T', type=float, default=120.0, help='Total time [h]')
    p.add_argument('--drought', type=float, default=48.0, help='Drought onset [h]')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', type=str, default=None, help='Save figure path')
    p.add_argument('--no-plot', action='store_true')
    return p.parse_args()


def single_run(args):
    """Fast single-run experiment."""
    params = SimulationParams(
        grid_res    = 40,
        dt          = 0.3,
        record_every= 3,
        N_init      = 150,
        max_nodes   = 800,
        domain_size = 5.0,
        eps         = 0.05,
    )
    print("Running single simulation (fast mode)...")
    print(f"  Domain: {params.domain_size}cm x {params.domain_size}cm")
    print(f"  eps = {params.eps} cm, D = {params.D:.2e} cm²/h")
    print(f"  T = {args.T} h, drought at t = {args.drought} h")
    print()

    sim = MycoNetSimulation(params=params, seed=args.seed)
    results = sim.run(T=args.T, drought_onset=args.drought,
                      drought_strength=0.85, verbose=True)

    print()
    results.summary()

    if not args.no_plot:
        results.plot(save_path=args.save)
    return results


def ensemble_run(args):
    """Full 10-run ensemble matching the paper."""
    params = SimulationParams(
        grid_res    = 40,
        dt          = 0.25,
        record_every= 4,
        N_init      = 150,
        max_nodes   = 800,
        domain_size = 5.0,
        eps         = 0.05,
    )
    print(f"Running {args.n_runs}-run ensemble...")
    print()

    ensemble = run_ensemble(
        n_runs          = args.n_runs,
        T               = args.T,
        drought_onset   = args.drought,
        drought_strength= 0.85,
        verbose         = False,
        params          = params,
    )

    # Print summary statistics
    t   = ensemble['time']
    sig = ensemble['sigma_r_mean']
    w2  = ensemble['w2_mean']
    psi = ensemble['psi_mean']
    pb  = ensemble['psi_bound_mean']

    drought_idx = np.searchsorted(t, args.drought)
    print(f"{'Quantity':<38} {'Pre-drought':>13} {'Post-drought':>13}")
    print("-" * 66)
    for name, arr in [
        ("sigma_r (Freiman index)", sig),
        ("W2/eps (Wasserstein)", w2),
        ("Psi*eps^2/D (dissipation)", psi),
        ("Psi lower bound (Thm 6.1)", pb),
    ]:
        pre  = arr[:drought_idx].mean()
        post = arr[drought_idx:].mean()
        print(f"{name:<38} {pre:>13.4f} {post:>13.4f}")

    # Quadratic scaling check
    ex_pre  = max(0, sig[:drought_idx].mean() - K_HEX)
    ex_post = max(0, sig[drought_idx:].mean() - K_HEX)
    if ex_pre > 0:
        ratio = (ex_post / ex_pre)**2
        psi_ratio = psi[drought_idx:].mean() / psi[:drought_idx].mean()
        print()
        print(f"Predicted Psi ratio via (sigma_r scaling)^2  = {ratio:.2f}")
        print(f"Observed  Psi ratio from simulation           = {psi_ratio:.2f}")

    # Empirical C_sim (slope of Psi vs (sigma_r - K_HEX)^2)
    ex2  = np.maximum(sig - K_HEX, 0)**2
    valid = ex2 > 1e-4
    if valid.sum() > 5:
        slope = np.polyfit(ex2[valid], psi[valid], 1)[0]
        print(f"\nEmpirical C_sim = {slope:.2f}  "
              f"(theory: C_STAR ≈ 21.8)")

    if not args.no_plot:
        _plot_ensemble(ensemble, args)

    return ensemble


def _plot_ensemble(ens, args):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t   = ens['time']
    sig = ens['sigma_r_mean']
    sig_sd = ens['sigma_r_std']
    w2  = ens['w2_mean']
    w2_sd = ens['w2_std']
    psi = ens['psi_mean']
    psi_sd = ens['psi_std']
    pb  = ens['psi_bound_mean']

    from myconet import K_HEX, c0
    drought = args.drought
    colors = {'sig': '#2a5c8a', 'w2': '#2a6a3a', 'psi': '#8a2a2a'}

    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(3, 1, hspace=0.40)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax in axes:
        ax.axvline(drought, color='grey', lw=1.2, ls='--', alpha=0.6)
        ax.set_xlim(t[0], t[-1])

    def fill_band(ax, arr, sd, color):
        ax.fill_between(t, arr - sd, arr + sd, alpha=0.18, color=color)

    # (a) Freiman index
    axes[0].axhline(K_HEX, color='grey', lw=0.9, ls=':', alpha=0.7)
    fill_band(axes[0], sig, sig_sd, colors['sig'])
    axes[0].plot(t, sig, color=colors['sig'], lw=1.8)
    axes[0].text(0.02, K_HEX + 0.05, r'$K_{\rm hex}=19/7$',
                 fontsize=8.5, color='grey', transform=axes[0].get_yaxis_transform())
    axes[0].set_ylabel(r'$\sigma_r(\Gamma)$', fontsize=12)
    axes[0].set_title('(a) Local Freiman Index', fontsize=10, loc='left')

    # (b) W2
    w2_lb = c0 * np.maximum(sig - K_HEX, 0)
    fill_band(axes[1], w2, w2_sd, colors['w2'])
    axes[1].plot(t, w2, color=colors['w2'], lw=1.8, label='Simulation')
    axes[1].plot(t, w2_lb, color=colors['w2'], lw=1.0, ls='--',
                 alpha=0.7, label='Prop. 4.4 lower bound')
    axes[1].legend(fontsize=8.5, framealpha=0.75, loc='upper left')
    axes[1].set_ylabel(r'$W_2/\varepsilon$', fontsize=12)
    axes[1].set_title('(b) Wasserstein Distance', fontsize=10, loc='left')

    # (c) Psi
    fill_band(axes[2], psi, psi_sd, colors['psi'])
    axes[2].plot(t, psi, color=colors['psi'], lw=1.8, label='Simulation')
    axes[2].plot(t, pb, color=colors['psi'], lw=1.0, ls='--',
                 alpha=0.7, label='Thm. 6.1 lower bound')
    axes[2].legend(fontsize=8.5, framealpha=0.75, loc='upper left')
    axes[2].set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=12)
    axes[2].set_xlabel('Time (h)', fontsize=11)
    axes[2].set_title('(c) Metabolic Dissipation', fontsize=10, loc='left')

    # Inset scatter
    ax_ins = axes[2].inset_axes([0.62, 0.10, 0.35, 0.52])
    ex2 = np.maximum(sig - K_HEX, 0)**2
    valid = ex2 > 1e-5
    ax_ins.scatter(ex2[valid], psi[valid], s=5, alpha=0.6, color=colors['psi'])
    if valid.sum() > 5:
        slope = np.polyfit(ex2[valid], psi[valid], 1)[0]
        xf = np.linspace(0, ex2.max(), 50)
        ax_ins.plot(xf, slope * xf, 'r-', lw=1.3)
        ax_ins.text(0.05, 0.88, f'$C_{{\\rm sim}}\\approx{slope:.1f}$',
                   transform=ax_ins.transAxes, fontsize=7.5, color='red')
    ax_ins.set_xlabel(r'$(\sigma_r-K_{{\rm hex}})^2$', fontsize=7.5)
    ax_ins.set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=7.5)
    ax_ins.tick_params(labelsize=6.5)

    plt.suptitle(
        f'MycoNet: Freiman–Villani Analysis ({args.n_runs}-run ensemble)',
        fontsize=11, y=0.998
    )

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {args.save}")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    if args.ensemble:
        ensemble_run(args)
    else:
        single_run(args)


def main():
    """Entry point for the myconet-drought console script."""
    args = parse_args()
    if args.ensemble:
        ensemble_run(args)
    else:
        single_run(args)
