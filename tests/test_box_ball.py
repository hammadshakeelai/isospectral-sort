"""Unit tests for Box-Ball System soliton sorting."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.box_ball import box_ball_sort, bbs_step, extract_soliton_lengths


def test_bbs_step_soliton_movement():
    # Soliton of length 3: [1, 1, 1, 0, 0, 0, 0, 0]
    lattice = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    next_lattice = bbs_step(lattice)
    # After 1 step, a soliton of length 3 moves 3 positions: [0, 0, 0, 1, 1, 1, 0, 0]
    assert np.array_equal(next_lattice, [0, 0, 0, 1, 1, 1, 0, 0])


def test_box_ball_sort_ascending():
    arr = [42.0, -10.0, 19.0, 3.0, 99.0, 12.0]
    sorted_arr, diag = box_ball_sort(arr, reverse=False, return_diagnostics=True)
    assert np.allclose(sorted_arr, sorted(arr))
    assert diag.time_steps > 0


def test_box_ball_sort_descending():
    arr = [5.0, 2.0, 8.0, 1.0]
    sorted_arr = box_ball_sort(arr, reverse=True)
    assert np.allclose(sorted_arr, sorted(arr, reverse=True))
