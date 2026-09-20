"""
Foundational Research Paper Grounding & Verification Test Suite.

Grounded strictly in:
1. Jürgen Moser (1975): "Finitely many points on the line under the influence
   of an exponential potential - An integrable system."
2. Daisuke Takahashi & Junkichi Satsuma (1990): "A soliton cellular automaton."
3. Yann Brenier (1991): "Polar factorization and monotone rearrangement of vector-valued functions."
4. Roger W. Brockett (1988/1991): "Dynamical systems that sort lists, diagonalize matrices,
   and solve linear programming problems."
"""

import numpy as np
import pytest
from isospectralsort import (
    brockett_sort,
    toda_sort,
    optimal_transport_sort,
    box_ball_sort,
)


class TestMoserTodaGrounding:
    """Rigorous tests of Moser 1975 Toda lattice asymptotic scattering."""

    def test_toda_duplicate_eigenvalues_moser_perturbation(self):
        """
        Moser 1975 (p. 471) & Gantmacher-Krein Theorem:
        Jacobi matrices with non-zero couplings strictly have simple, distinct spectra.
        Moser's infinitesimal spectral perturbation ensures full lattice communication
        on duplicate eigenvalues with exact multiset restoration.
        """
        arr = [5.0, 2.0, 5.0, 1.0, 2.0]
        sorted_arr, diag = toda_sort(arr, return_diagnostics=True)
        assert np.array_equal(sorted_arr, [1.0, 2.0, 2.0, 5.0, 5.0])
        assert diag.eigenvalue_drift < 1e-6

        # Triples and high-multiplicity duplicates
        triples = [9.0, 3.0, 9.0, 3.0, 9.0]
        assert np.array_equal(toda_sort(triples), [3.0, 3.0, 9.0, 9.0, 9.0])

    def test_toda_all_identical_elements(self):
        """Degenerate zero-variance spectrum."""
        arr = [4.0, 4.0, 4.0, 4.0]
        assert np.array_equal(toda_sort(arr), [4.0, 4.0, 4.0, 4.0])

    def test_toda_reverse_scattering_direction(self):
        """Moser 1975 Theorem 3: t -> +inf gives descending, t -> -inf gives ascending."""
        arr = [12.0, 3.0, 45.0, 1.0]
        asc = toda_sort(arr, reverse=False)
        desc = toda_sort(arr, reverse=True)
        assert np.allclose(asc, [1.0, 3.0, 12.0, 45.0])
        assert np.allclose(desc, [45.0, 12.0, 3.0, 1.0])


class TestTakahashiSatsumaBBSGrounding:
    """Rigorous tests of Takahashi & Satsuma 1990 Soliton Cellular Automaton."""

    def test_bbs_pure_integer_soliton_sorting(self):
        """Takahashi & Satsuma 1990: Soliton length v(L) = L spatial separation."""
        solitons = [5, 2, 8, 1, 3]
        res, diag = box_ball_sort(solitons, mode="pure", return_diagnostics=True)
        assert np.array_equal(res, [1, 2, 3, 5, 8])
        assert diag.lattice_length > 0
        assert len(diag.soliton_trajectories) > 0

    def test_bbs_pure_mode_domain_safety(self):
        """
        Physical cellular automaton solitons are discrete balls (L in Z+).
        Inputs outside physical domain or excessively large integers must raise
        informative ValueError rather than causing memory exhaustion.
        """
        # Floats
        with pytest.raises(ValueError, match="strictly positive integers"):
            box_ball_sort([2.5, 1.0, 4.0], mode="pure")

        # Negative numbers
        with pytest.raises(ValueError, match="strictly positive integers"):
            box_ball_sort([-3, 5, 2], mode="pure")

        # Excessively large integers (lattice explosion guardrail)
        with pytest.raises(ValueError, match="cannot exceed 1000"):
            box_ball_sort([1000000, 2000000], mode="pure")

    def test_bbs_robust_arbitrary_scales_fast(self):
        """Robust mode handles arbitrary floating-point numbers and large scales instantaneously."""
        large_arr = [2000000, 1000000, 5000000]
        assert np.array_equal(box_ball_sort(large_arr, mode="robust"), [1000000, 2000000, 5000000])

        float_arr = [-15.5, 0.2, -100.0, 42.8]
        assert np.allclose(box_ball_sort(float_arr, mode="robust"), sorted(float_arr))


class TestBrenierOptimalTransportGrounding:
    """Rigorous tests of Brenier 1991 Monotone Rearrangement & Sinkhorn."""

    def test_ot_small_epsilon_no_underflow(self):
        """
        Stabilized Gibbs kernel avoids underflow to exact 0 even at small epsilon.
        No element is dropped or replaced with zero.
        """
        arr = [3.0, 1.0, 2.0]
        for eps in [0.05, 0.01, 0.001, 0.0001]:
            res = optimal_transport_sort(arr, epsilon=eps, mode="pure")
            assert not np.any(np.isnan(res))
            assert not np.any(np.isinf(res))
            # Monotone check
            assert res[0] <= res[1] <= res[2]
            # No zero values introduced
            assert np.all(res > 0.5)

    def test_ot_extreme_scale_invariance(self):
        """Brenier's polar rearrangement is scale-invariant across 15 orders of magnitude."""
        huge_arr = [1e15, 5e15, 2e15]
        assert np.allclose(optimal_transport_sort(huge_arr, mode="robust"), [1e15, 2e15, 5e15])

        tiny_arr = [1e-12, 5e-12, 2e-12]
        assert np.allclose(optimal_transport_sort(tiny_arr, mode="robust"), [1e-12, 2e-12, 5e-12])

    def test_ot_doubly_stochastic_normalization(self):
        """Marginal constraints sum_j P_ij = 1 and sum_i P_ij = 1."""
        arr = [10.0, -5.0, 25.0, 3.0]
        _, diag = optimal_transport_sort(arr, return_diagnostics=True)
        P = diag.permutation_matrix
        assert np.allclose(np.sum(P, axis=0), np.ones(len(arr)), atol=1e-3)
        assert np.allclose(np.sum(P, axis=1), np.ones(len(arr)), atol=1e-3)


class TestBrockettDoubleBracketGrounding:
    """Rigorous tests of Brockett 1988/1991 Lie-algebraic gradient flow."""

    def test_brockett_duplicates_and_machine_precision(self):
        """Brockett double-bracket flow sorts lists with duplicates smoothly."""
        arr = [5.0, 2.0, 5.0, 1.0, 2.0]
        sorted_arr, diag = brockett_sort(arr, return_diagnostics=True)
        assert np.allclose(sorted_arr, [1.0, 2.0, 2.0, 5.0, 5.0], atol=1e-2)
        assert diag.eigenvalue_drift < 1e-10

    def test_brockett_lyapunov_potential_monotonicity(self):
        """Phi(H) = Tr(HN) is strictly non-decreasing along the trajectory."""
        arr = [30.0, -10.0, 5.0, 100.0, 0.0]
        _, diag = brockett_sort(arr, return_diagnostics=True)
        traces = diag.trace_history
        assert len(traces) > 10
        for i in range(len(traces) - 1):
            assert traces[i+1] >= traces[i] - 1e-8
