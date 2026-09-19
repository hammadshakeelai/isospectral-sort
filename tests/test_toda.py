"""Unit tests for Toda lattice Lax pair sorting."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.toda import toda_sort


def test_toda_sort_ascending():
    arr = [25.0, 3.0, 18.0, 1.0, 50.0]
    sorted_arr, diag = toda_sort(arr, reverse=False, return_diagnostics=True)
    expected = sorted(arr)
    assert np.allclose(sorted_arr, expected, atol=0.5)
    assert diag.eigenvalue_drift < 1e-10


def test_toda_sort_descending():
    arr = [25.0, 3.0, 18.0, 1.0, 50.0]
    sorted_arr, diag = toda_sort(arr, reverse=True, return_diagnostics=True)
    expected = sorted(arr, reverse=True)
    assert np.allclose(sorted_arr, expected, atol=0.5)
    assert diag.eigenvalue_drift < 1e-10


def test_toda_degenerate():
    assert len(toda_sort([])) == 0
    assert np.allclose(toda_sort([5.0]), [5.0])
