"""Unit tests for Brockett double-bracket flow sorting."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.brockett import brockett_sort


def test_brockett_sort_ascending():
    arr = [42.0, 7.0, 19.0, 3.0, 99.0, 12.0]
    sorted_arr, diag = brockett_sort(arr, reverse=False, return_diagnostics=True)
    expected = sorted(arr)
    assert np.allclose(sorted_arr, expected, atol=1e-2)
    assert diag.converged is True
    assert diag.eigenvalue_drift < 1e-10


def test_brockett_sort_descending():
    arr = [42.0, 7.0, 19.0, 3.0, 99.0, 12.0]
    sorted_arr, diag = brockett_sort(arr, reverse=True, return_diagnostics=True)
    expected = sorted(arr, reverse=True)
    assert np.allclose(sorted_arr, expected, atol=1e-2)
    assert diag.converged is True


def test_brockett_degenerate_cases():
    # Empty
    res_empty = brockett_sort([])
    assert len(res_empty) == 0

    # Single element
    res_single = brockett_sort([42.0])
    assert np.allclose(res_single, [42.0])

    # Two elements
    res_two = brockett_sort([10.0, 2.0])
    assert np.allclose(res_two, [2.0, 10.0], atol=1e-2)


def test_brockett_negative_numbers():
    arr = [-15.0, 3.0, -100.0, 0.0, 25.0]
    sorted_arr = brockett_sort(arr)
    expected = sorted(arr)
    assert np.allclose(sorted_arr, expected, atol=1e-2)
