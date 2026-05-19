"""tests/test_freiman.py"""
import numpy as np
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from myconet.freiman import (local_freiman_index, freiman_excess,
                             dissipation_lower_bound, K_HEX, C_STAR, c0)
from myconet.network import hexagonal_lattice

class TestConstants:
    def test_K_HEX(self):   assert abs(K_HEX - 19/7) < 1e-10
    def test_c0(self):      assert abs(c0 - 7/24) < 1e-10
    def test_C_STAR(self):  assert abs(C_STAR - 256*49/576) < 1e-8

class TestHexLattice:
    def test_K_HEX_recovered(self):
        """Interior hex lattice nodes recover K_HEX = 19/7 (Lemma 4.1)."""
        eps = 0.1; L = 4.0
        n   = hexagonal_lattice(eps, L)
        interior = n[(n[:,0]>6*eps)&(n[:,0]<L-6*eps)&
                     (n[:,1]>6*eps)&(n[:,1]<L-6*eps)]
        sig, vals = local_freiman_index(interior, eps, k=7)
        assert abs(np.median(vals) - K_HEX) < 0.05, \
            f"Median={np.median(vals):.4f}, expected {K_HEX:.4f}"

class TestPerturbative:
    def _lattice(self, eps, L=2.0):
        n = hexagonal_lattice(eps, L)
        return n[(n[:,0]>5*eps)&(n[:,0]<L-5*eps)&
                 (n[:,1]>5*eps)&(n[:,1]<L-5*eps)]

    def test_upper_bound(self):
        """Lemma 4.2 upper bound holds."""
        eps, d = 0.1, 0.03
        rng = np.random.default_rng(7)
        nodes = self._lattice(eps) + rng.uniform(-d, d, self._lattice(eps).shape)
        sig, _ = local_freiman_index(nodes, eps)
        assert sig <= K_HEX + (24/7)*(d/eps) + 0.2

    def test_excess_near_zero_for_hex(self):
        eps = 0.1
        nodes = self._lattice(eps)
        excess = freiman_excess(nodes, eps)
        assert abs(excess) < 0.1, f"Hex excess={excess:.4f}"

    def test_large_displacement_raises_sigma(self):
        """Large displacements raise sigma_r above the hexagonal baseline."""
        eps = 0.1
        rng = np.random.default_rng(13)
        base = self._lattice(eps)
        small = base + rng.uniform(-0.005, 0.005, base.shape)
        large = base + rng.uniform(-0.04,  0.04,  base.shape)
        sig_s, _ = local_freiman_index(small, eps)
        sig_l, _ = local_freiman_index(large, eps)
        assert sig_l > sig_s, f"Large({sig_l:.3f}) <= Small({sig_s:.3f})"

class TestDissipationBound:
    def test_zero_at_K_HEX(self):
        assert dissipation_lower_bound(K_HEX, 0.01, 3.6e-3) == 0.0
    def test_quadratic(self):
        eps, D = 0.01, 3.6e-3
        p1 = dissipation_lower_bound(K_HEX+0.5, eps, D)
        p2 = dissipation_lower_bound(K_HEX+1.0, eps, D)
        assert abs(p2/p1 - 4.0) < 1e-8
    def test_monotone(self):
        eps, D = 0.01, 3.6e-3
        v = [dissipation_lower_bound(K_HEX+ex, eps, D) for ex in [0,.1,.5,1,2]]
        assert all(v[i] <= v[i+1] for i in range(len(v)-1))

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
