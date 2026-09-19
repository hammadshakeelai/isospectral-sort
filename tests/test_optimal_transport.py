"""Unit tests for Optimal Transport & Sinkhorn sorting."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.optimal_transport import optimal_transport_sort


def test_ot_sort_hard():
    arr = [42.0, -10.0, 19.0, 3.0, 99.0, 12.0]
    sorted_asc, diag = optimal_transport_sort(arr, reverse=False, return_diagnostics=True)
    assert np.allclose(sorted_asc, sorted(arr))
    assert diag.converged is True

    sorted_desc = optimal_transport_sort(arr, reverse=True)
    assert np.allclose(sorted_desc, sorted(arr, reverse=True))


def test_ot_doubly_stochastic():
    arr = [5.0, 1.0, 8.0, 3.0]
    _, diag = optimal_transport_sort(arr, return_diagnostics=True)
    P = diag.permutation_matrix
    # Check row sums and col sums are 1
    assert np.allclose(np.sum(P, axis=0), np.ones(len(arr)), atol=1e-3)
    assert np.allclose(np.sum(P, axis=1), np.ones(len(arr)), atol=1e-3)


def test_ot_soft_differentiable():
    arr = [10.0, 2.0, 8.0, 4.0]
    soft = optimal_transport_sort(arr, soft=True)
    # Monotonically increasing check
    assert all(soft[i] <= soft[i+1] for i in range(len(soft) - 1))
