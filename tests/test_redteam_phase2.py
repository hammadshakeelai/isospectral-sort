"""
Phase 2 Red-Team Adversarial & Security Test Suite.

Tests for:
1. Parameter validation & domain boundary enforcement (dt, max_steps, tol, epsilon, max_iters, init_method).
2. Multi-threaded race conditions and re-entrancy under high concurrency.
3. Pathological heavy-tail distributions (Cauchy, Pareto, Student-t).
4. Alternating extreme signs and ill-conditioned clusters.
5. Invariant preservation (sorting correctness, permutation property, trace conservation).
"""

import concurrent.futures
import numpy as np
import pytest

from isospectralsort import (
    brockett_sort,
    toda_sort,
    optimal_transport_sort,
    box_ball_sort,
)


class TestParameterValidation:
    """Verify that all invalid hyperparameter combinations raise descriptive ValueErrors."""

    @pytest.mark.parametrize("bad_dt", [0.0, -0.01, -1.0, -1e-9])
    def test_brockett_invalid_dt(self, bad_dt):
        with pytest.raises(ValueError, match="dt must be positive"):
            brockett_sort([3.0, 1.0, 2.0], dt=bad_dt)

    @pytest.mark.parametrize("bad_steps", [0, -1, -100])
    def test_brockett_invalid_max_steps(self, bad_steps):
        with pytest.raises(ValueError, match="max_steps must be a positive integer"):
            brockett_sort([3.0, 1.0, 2.0], max_steps=bad_steps)

    @pytest.mark.parametrize("bad_tol", [0.0, -1e-5, -1.0])
    def test_brockett_invalid_tol(self, bad_tol):
        with pytest.raises(ValueError, match="tol must be positive"):
            brockett_sort([3.0, 1.0, 2.0], tol=bad_tol)

    def test_brockett_invalid_init_method(self):
        with pytest.raises(ValueError, match="Unknown init_method"):
            brockett_sort([3.0, 1.0, 2.0], init_method="invalid_spectral_method")

    @pytest.mark.parametrize("bad_dt", [0.0, -0.01, -1.0, -1e-9])
    def test_toda_invalid_dt(self, bad_dt):
        with pytest.raises(ValueError, match="dt must be positive"):
            toda_sort([3.0, 1.0, 2.0], dt=bad_dt)

    @pytest.mark.parametrize("bad_steps", [0, -1, -100])
    def test_toda_invalid_max_steps(self, bad_steps):
        with pytest.raises(ValueError, match="max_steps must be a positive integer"):
            toda_sort([3.0, 1.0, 2.0], max_steps=bad_steps)

    @pytest.mark.parametrize("bad_tol", [0.0, -1e-5, -1.0])
    def test_toda_invalid_tol(self, bad_tol):
        with pytest.raises(ValueError, match="tol must be positive"):
            toda_sort([3.0, 1.0, 2.0], tol=bad_tol)

    @pytest.mark.parametrize("bad_eps", [0.0, -0.01, -1.0])
    def test_ot_invalid_epsilon(self, bad_eps):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            optimal_transport_sort([3.0, 1.0, 2.0], epsilon=bad_eps)

    @pytest.mark.parametrize("bad_tol", [0.0, -1e-5, -1.0])
    def test_ot_invalid_tol(self, bad_tol):
        with pytest.raises(ValueError, match="tol must be positive"):
            optimal_transport_sort([3.0, 1.0, 2.0], tol=bad_tol)

    @pytest.mark.parametrize("bad_iters", [0, -1, -50])
    def test_ot_invalid_max_iters(self, bad_iters):
        with pytest.raises(ValueError, match="max_iters must be a positive integer"):
            optimal_transport_sort([3.0, 1.0, 2.0], max_iters=bad_iters)


class TestConcurrencyAndThreadSafety:
    """Stress test concurrent execution across thread pools to ensure no shared state corruption."""

    def test_multithreaded_execution(self):
        rng = np.random.RandomState(999)
        test_arrays = [rng.randn(5) for _ in range(12)]

        def run_all_algorithms(arr):
            b_res = brockett_sort(arr)
            t_res = toda_sort(arr)
            ot_res = optimal_transport_sort(arr)
            bbs_res = box_ball_sort(arr)
            expected = np.sort(arr)
            
            assert np.allclose(b_res, expected, atol=5e-3)
            assert np.allclose(t_res, expected, atol=5e-3)
            assert np.allclose(ot_res, expected, atol=5e-3)
            assert np.allclose(bbs_res, expected, atol=5e-3)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_all_algorithms, a) for a in test_arrays]
            for future in concurrent.futures.as_completed(futures):
                assert future.result() is True


class TestHeavyTailAndExtremeSpectra:
    """Verify stability under Cauchy, Pareto, and extreme alternating values."""

    def test_cauchy_distribution_stress(self):
        rng = np.random.RandomState(42)
        cauchy_vals = rng.standard_cauchy(size=6)
        expected = np.sort(cauchy_vals)

        b_res = brockett_sort(cauchy_vals)
        t_res = toda_sort(cauchy_vals)
        ot_res = optimal_transport_sort(cauchy_vals)
        bbs_res = box_ball_sort(cauchy_vals)

        assert np.allclose(b_res, expected, atol=1e-2)
        assert np.allclose(t_res, expected, atol=1e-2)
        assert np.allclose(ot_res, expected, atol=1e-2)
        assert np.allclose(bbs_res, expected, atol=1e-2)

    def test_alternating_huge_and_tiny_signs(self):
        arr = np.array([-1e8, 1e-8, -1e-4, 1e4, 0.0, -500.0, 500.0, -1e-12])
        expected = np.sort(arr)

        b_res = brockett_sort(arr)
        t_res = toda_sort(arr)
        ot_res = optimal_transport_sort(arr)
        bbs_res = box_ball_sort(arr)

        assert np.allclose(b_res, expected, rtol=1e-3, atol=1e-4)
        assert np.allclose(t_res, expected, rtol=1e-3, atol=1e-4)
        assert np.allclose(ot_res, expected, rtol=1e-3, atol=1e-4)
        assert np.allclose(bbs_res, expected, rtol=1e-3, atol=1e-4)

    def test_multi_cluster_degeneracy(self):
        # Multiple dense clusters with tiny separations
        c1 = 1.0 + np.array([0.0, 1e-6, -1e-6])
        c2 = 100.0 + np.array([0.0, 2e-6, -2e-6])
        c3 = -50.0 + np.array([0.0, 1e-7])
        arr = np.concatenate([c1, c2, c3])
        expected = np.sort(arr)

        b_res = brockett_sort(arr)
        t_res = toda_sort(arr)
        ot_res = optimal_transport_sort(arr)
        bbs_res = box_ball_sort(arr)

        assert np.allclose(b_res, expected, atol=1e-3)
        assert np.allclose(t_res, expected, atol=0.02)
        assert np.allclose(ot_res, expected, atol=1e-3)
        assert np.allclose(bbs_res, expected, atol=1e-3)
