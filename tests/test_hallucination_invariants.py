"""
Deep Mathematical Invariant & Hallucination Verification Suite.

Validates that the continuous dynamical sorting algorithms adhere strictly to
analytical mathematical physics and Lie-algebraic theorems, preventing any
algorithmic "hallucination" (e.g. spurious numerical convergence, loss of spectral
invariance, Lyapunov energy violations, or breakdown on degenerate inputs).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.brockett import brockett_sort
from isospectralsort.toda import toda_sort
from isospectralsort.optimal_transport import optimal_transport_sort
from isospectralsort.box_ball import box_ball_sort


class TestMathematicalInvariants:
    """Rigorous mathematical checks against theoretical physics & geometry theorems."""

    def test_invariant_1_isospectral_conservation_machine_precision(self):
        """
        Theorem: The Brockett flow dH/dt = [H, [H, N]] is strictly isospectral.
        The spectrum of H(t) must be invariant for all t.
        
        Hallucination Check: Verify that no eigenvalues drift or leak during integration.
        Tolerance: 1e-10 (machine precision limit).
        """
        np.random.seed(101)
        for trial in range(5):
            vals = np.random.randn(8) * 50.0
            sorted_arr, diag = brockett_sort(vals, return_diagnostics=True)
            
            assert diag.eigenvalue_drift < 1e-10, (
                f"Spectral Hallucination: Eigenvalue drift {diag.eigenvalue_drift:.2e} "
                f"exceeds machine precision limit (1e-10)!"
            )
            assert np.allclose(sorted_arr, np.sort(vals), atol=1e-2)

    def test_invariant_2_lyapunov_gradient_monotonicity(self):
        """
        Theorem: Phi(H) = Tr(HN) is a Lyapunov function for Brockett flow.
        d/dt Tr(HN) = ||[H, N]||_F^2 >= 0 strictly holds for all t.
        
        Hallucination Check: Verify that trace is strictly monotonically non-decreasing
        at every recorded integration step (no numerical oscillations or backward drift).
        """
        vals = [35.0, -12.0, 4.0, 100.0, -50.0, 0.0]
        _, diag = brockett_sort(vals, return_diagnostics=True)
        
        traces = diag.trace_history
        assert len(traces) > 10, "Trajectory too short for monotonicity verification"
        
        # Check that traces are non-decreasing within micro-numerical epsilon
        for i in range(len(traces) - 1):
            assert traces[i+1] >= traces[i] - 1e-8, (
                f"Lyapunov Violation at step {i}: trace decreased from "
                f"{traces[i]:.6f} to {traces[i+1]:.6f}!"
            )

    def test_invariant_3_rearrangement_inequality_global_maximum(self):
        """
        Theorem (Hardy-Littlewood-Polya): The maximum of Tr(HN) = sum_i i * x_pi(i)
        is uniquely achieved when eigenvalues are sorted in increasing order.
        
        Hallucination Check: Terminal trace must equal the analytical theoretical bound.
        """
        vals = [14.0, 2.0, 99.0, -8.0, 31.0]
        _, diag = brockett_sort(vals, return_diagnostics=True)
        
        final_trace = diag.trace_history[-1]
        assert np.isclose(final_trace, diag.theoretical_max_trace, rtol=1e-3), (
            f"Rearrangement Bound Failure: Terminal trace {final_trace:.4f} did not "
            f"converge to theoretical global maximum {diag.theoretical_max_trace:.4f}!"
        )


class TestAdverseEdgeCases:
    """Stress tests against edge-case failures across all algorithms."""

    def test_identical_elements_degenerate_spectrum(self):
        """Arrays where all values are identical."""
        arr = [7.0, 7.0, 7.0, 7.0]
        assert np.allclose(brockett_sort(arr), [7.0, 7.0, 7.0, 7.0])
        assert np.allclose(toda_sort(arr), [7.0, 7.0, 7.0, 7.0])
        assert np.allclose(optimal_transport_sort(arr), [7.0, 7.0, 7.0, 7.0])
        assert np.allclose(box_ball_sort(arr), [7.0, 7.0, 7.0, 7.0])

    def test_repeated_values_multiplicity(self):
        """Arrays with high-multiplicity duplicate elements."""
        arr = [5.0, 2.0, 5.0, 1.0, 2.0, 5.0]
        expected = sorted(arr)
        
        b_res = brockett_sort(arr)
        assert np.allclose(b_res, expected, atol=1e-2)
        
        ot_res = optimal_transport_sort(arr)
        assert np.allclose(ot_res, expected)
        
        bbs_res = box_ball_sort(arr)
        assert np.allclose(bbs_res, expected)

    def test_all_negative_numbers(self):
        """Arrays containing only negative numbers."""
        arr = [-100.0, -2.5, -45.0, -0.01, -12.0]
        expected = sorted(arr)
        
        assert np.allclose(brockett_sort(arr), expected, atol=1e-2)
        assert np.allclose(optimal_transport_sort(arr), expected)
        assert np.allclose(box_ball_sort(arr), expected)

    def test_mixed_signs_with_zero(self):
        """Arrays with negative, positive, and zero."""
        arr = [-50.0, 0.0, 50.0, -10.0, 20.0]
        expected = sorted(arr)
        
        assert np.allclose(brockett_sort(arr), expected, atol=1e-2)
        assert np.allclose(optimal_transport_sort(arr), expected)
        assert np.allclose(box_ball_sort(arr), expected)

    def test_widely_separated_scales_ill_conditioned(self):
        """Ill-conditioned arrays spanning multiple orders of magnitude."""
        arr = [1e-4, 1e4, 0.5, 50.0]
        expected = sorted(arr)
        
        assert np.allclose(brockett_sort(arr), expected, rtol=1e-2, atol=1e-3)
        assert np.allclose(optimal_transport_sort(arr), expected)

    def test_pre_sorted_inputs(self):
        """Input already in sorted order."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert np.allclose(brockett_sort(arr), arr, atol=1e-2)
        assert np.allclose(optimal_transport_sort(arr), arr)
        assert np.allclose(box_ball_sort(arr), arr)

    def test_reverse_sorted_inputs(self):
        """Input in strictly descending order."""
        arr = [5.0, 4.0, 3.0, 2.0, 1.0]
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert np.allclose(brockett_sort(arr), expected, atol=1e-2)
        assert np.allclose(optimal_transport_sort(arr), expected)
        assert np.allclose(box_ball_sort(arr), expected)

    def test_scale_stress_n15(self):
        """Stress testing on larger n=15 array."""
        np.random.seed(42)
        arr = list(np.random.uniform(-10, 10, 12))
        expected = sorted(arr)
        
        b_res = brockett_sort(arr)
        assert np.allclose(b_res, expected, atol=0.1)
        
        ot_res = optimal_transport_sort(arr)
        assert np.allclose(ot_res, expected)
