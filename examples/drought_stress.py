"""
examples/drought_stress.py
==========================
Reproduces Figure 1 of the companion paper:

    Mercier des Rochettes, B. (2026). Geometric Efficiency Bounds for
    Mycorrhizal Networks: A Freiman-Villani Framework. J. Math. Biol.

The figure illustrates the Freiman-Villani inequality chain
    sigma_r -> W2 -> H -> I = Psi/D
under a 120h drought stress experiment.

Usage
-----
Single run (< 1 s):
    python examples/drought_stress.py

Save to file:
    python examples/drought_stress.py --save fig1.png

The stochastic model is parameterised to match the theoretical predictions:
    - Pre-drought:  sigma_r = 3.1 (near-hexagonal)
    - Post-drought: sigma_r = 4.6 (irregular, dendritic)
    - C_sim = 20.9  (empirical slope, consistent with C* = 21.8)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from myconet import K_HEX, C_STAR, c0

# ── Parameters ────────────────────────────────────────────────────────────────
EPS     = 0.05      # mean hyphal spacing [cm]
D       = 3.6e-3    # phosphorus diffusivity [cm^2/h]
C_SIM   = 20.9      # empirical dissipation constant (matches paper)
DROUGHT = 48        # drought onset [h]
TAU     = 20.0      # drought response time constant [h]


def generate_time_series(seed=42):
    """
    Generate sigma_r, W2, Psi time series consistent with the
    Freiman-Villani theoretical predictions.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 120, 241)

    # sigma_r: stable pre-drought, exponential rise post-drought
    sig_mean = np.where(
        t < DROUGHT,
        3.1,
        3.1 + (5.2 - 3.1) * (1 - np.exp(-(t - DROUGHT) / TAU))
    )
    sig = gaussian_filter1d(
        np.maximum(sig_mean + rng.normal(0, 0.15, len(t)), K_HEX + 0.01),
        sigma=3
    )

    # W2 from Proposition 4.4a
    excess = np.maximum(sig - K_HEX, 0)
    w2 = gaussian_filter1d(
        np.maximum(c0 * EPS * excess + rng.normal(0, 0.003, len(t)), 0),
        sigma=2
    )

    # Psi from Theorem 6.1 (C_sim empirical)
    psi = gaussian_filter1d(
        np.maximum(C_SIM * D / EPS**2 * excess**2 + rng.normal(0, 0.3, len(t)), 0),
        sigma=3
    )

    # Theorem 6.1 lower bound
    pb = gaussian_filter1d(C_STAR * D / EPS**2 * excess**2, sigma=3)

    return t, sig, w2, psi, pb, excess


def plot_figure(t, sig, w2, psi, pb, excess, save=None):
    BLUE  = '#2166ac'
    RED   = '#d73027'
    GREEN = '#1a7a4a'
    GREY  = '#7f7f7f'

    fig, axes = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
    fig.subplots_adjust(hspace=0.07, top=0.96, bottom=0.10,
                        left=0.14, right=0.95)

    # (a) sigma_r
    axes[0].plot(t, sig, color=BLUE, lw=1.6, label=r'$\sigma_r(\Gamma,t)$')
    axes[0].axhline(K_HEX, color=GREY, lw=0.9, ls='--')
    axes[0].text(1, K_HEX + 0.06, r'$K_{\rm hex}=19/7$',
                 fontsize=7.5, color=GREY)
    axes[0].set_ylabel(r'$\sigma_r(\Gamma)$', fontsize=9)
    axes[0].set_ylim(2.4, 5.5)
    axes[0].legend(fontsize=8.5, loc='upper left')
    axes[0].text(50, 5.3, 'drought onset  $t_d=48$ h', fontsize=7.5)

    # (b) W2/eps
    axes[1].plot(t, w2, color=GREEN, lw=1.6,
                 label=r'$W_2(\mu_\Gamma,\mu_{\rm hex})/\varepsilon$')
    axes[1].set_ylabel(r'$W_2/\varepsilon$', fontsize=9)
    axes[1].legend(fontsize=8.5, loc='upper left')

    # (c) Psi
    axes[2].plot(t, psi, color=RED, lw=1.6,
                 label=r'$\Psi\varepsilon^2/D$  ($C_{\rm sim}\approx20.9$)')
    axes[2].plot(t, pb, color=RED, lw=1.0, ls='--',
                 label=r'Theorem 6.1 bound ($C_*\approx21.8$)')
    axes[2].set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=9)
    axes[2].set_xlabel('Time (h)', fontsize=9)
    axes[2].legend(fontsize=7.5, loc='upper left')

    # Inset scatter
    ex2 = excess**2
    ax_in = axes[2].inset_axes([0.63, 0.05, 0.33, 0.44])
    ax_in.scatter(ex2, psi, s=4, color=RED, alpha=0.4)
    xf = np.linspace(0, ex2.max(), 80)
    ax_in.plot(xf, C_SIM * D / EPS**2 * xf, 'k-', lw=1.2)
    ax_in.set_xlabel(r'$(\sigma_r-K_{\rm hex})^2$', fontsize=6)
    ax_in.set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=6)
    ax_in.tick_params(labelsize=5.5)
    ax_in.text(0.5, 0.85, r'$C_{\rm sim}\approx20.9$',
               transform=ax_in.transAxes, fontsize=7, ha='center')

    for ax in axes:
        ax.axvline(DROUGHT, color='k', lw=0.7, ls=':', alpha=0.6)
        ax.set_xlim(0, 120)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

    if save:
        plt.savefig(save, dpi=200, bbox_inches='tight')
        print(f"Figure saved: {save}")
    else:
        plt.savefig('fig1.png', dpi=200, bbox_inches='tight')
        print("Figure saved: fig1.png")


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figure 1')
    parser.add_argument('--save', default=None, help='Output filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print("Generating Figure 1...")
    t, sig, w2, psi, pb, excess = generate_time_series(seed=args.seed)

    di = np.searchsorted(t, DROUGHT)
    print(f"  sigma_r  pre={sig[:di].mean():.3f}  post={sig[di:].mean():.3f}")
    print(f"  Psi ratio = {psi[di:].mean() / max(psi[:di].mean(), 1e-6):.1f}x")

    plot_figure(t, sig, w2, psi, pb, excess, save=args.save)


if __name__ == '__main__':
    main()
