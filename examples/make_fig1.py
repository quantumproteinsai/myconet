"""
make_fig1.py  —  run from the myconet folder
Usage:  python make_fig1.py
Output: fig1.png  (in the same folder)
"""
import sys, os
# Add the myconet package folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from myconet import MycoNetSimulation, SimulationParams, K_HEX, C_STAR

print("Running simulation (single run, ~3 minutes)...")
params = SimulationParams(
    domain_size=5.0, grid_res=40, eps=0.05,
    dt=0.5, record_every=4, N_init=150,
)
sim = MycoNetSimulation(params=params, seed=42)
r   = sim.run(T=120, drought_onset=48, drought_strength=0.85, verbose=True)
print("Simulation done.")

t   = r.time
sig = r.sigma_r
w2  = r.w2_eps
psi = gaussian_filter1d(r.psi,       sigma=4)   # smooth noise
pb  = gaussian_filter1d(r.psi_bound, sigma=4)

ex2   = np.maximum(sig - K_HEX, 0)**2
valid = ex2 > 1e-4
C_sim = float(np.polyfit(ex2[valid], psi[valid], 1)[0])
print(f"C_sim = {C_sim:.2f}  (theory C* ≈ 21.8)")

# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE = '#2166ac'; RED = '#d73027'; GREEN = '#1a7a4a'; GREY = '#7f7f7f'

fig, axes = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
fig.subplots_adjust(hspace=0.07, top=0.96, bottom=0.10, left=0.14, right=0.95)

# (a) sigma_r
axes[0].plot(t, sig, color=BLUE, lw=1.6, label=r'$\sigma_r(\Gamma,t)$')
axes[0].axhline(K_HEX, color=GREY, lw=0.9, ls='--')
axes[0].text(1, K_HEX+0.06, r'$K_{\rm hex}=19/7$', fontsize=8, color=GREY)
axes[0].set_ylabel(r'$\sigma_r(\Gamma)$', fontsize=9)
axes[0].set_ylim(2.4, 5.4)
axes[0].legend(fontsize=9, loc='upper left')
axes[0].text(50, 5.1, 'drought onset  $t_d = 48$ h', fontsize=8)

# (b) W2/eps
axes[1].plot(t, w2, color=GREEN, lw=1.6,
             label=r'$W_2(\mu_\Gamma,\mu_{\rm hex})/\varepsilon$')
axes[1].set_ylabel(r'$W_2/\varepsilon$', fontsize=9)
axes[1].legend(fontsize=9, loc='upper left')

# (c) Psi (smoothed)
axes[2].plot(t, psi, color=RED, lw=1.6, label=r'$\Psi\varepsilon^2/D$')
axes[2].plot(t, pb,  color=RED, lw=1.0, ls='--',
             label=r'Theorem 6.1 bound ($C_*\approx21.8$)')
axes[2].set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=9)
axes[2].set_xlabel('Time (h)', fontsize=9)
axes[2].legend(fontsize=8, loc='upper left')

# Inset: Psi vs (sigma_r - K_hex)^2
ax_in = axes[2].inset_axes([0.63, 0.50, 0.33, 0.44])
ax_in.scatter(ex2[valid], psi[valid], s=6, color=RED, alpha=0.5)
xf = np.linspace(0, ex2[valid].max(), 80)
ax_in.plot(xf, C_sim * xf, 'k-', lw=1.2)
ax_in.set_xlabel(r'$(\sigma_r-K_{\rm hex})^2$', fontsize=6)
ax_in.set_ylabel(r'$\Psi\varepsilon^2/D$', fontsize=6)
ax_in.tick_params(labelsize=5.5)
ax_in.text(0.06, 0.85, f'$C_{{\\rm sim}}\\approx{C_sim:.1f}$',
           transform=ax_in.transAxes, fontsize=7)

for ax in axes:
    ax.axvline(48, color='k', lw=0.7, ls=':', alpha=0.6)
    ax.set_xlim(0, 120)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out}")
