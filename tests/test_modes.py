"""
Tests for explicit algorithmic modes: mode='pure' vs mode='robust'.
"""

import numpy as np
import pytest
from isospectralsort import (
    brockett_sort,
    toda_sort,
    optimal_transport_sort,
    box_ball_sort,
)


def test_pure_mode_monotonic_order():
    """Verify that pure mode produces sorted sequences on well-scaled data."""
    arr = [24.0, 11.0, 89.0, -5.0, 42.0, 0.0]
    expected = sorted(arr)

    # 1. Brockett pure mode
    b_sorted = brockett_sort(arr, mode="pure")
    assert np.allclose(b_sorted, expected, atol=1e-2)

    # 2. Toda pure mode (asymptotic flow)
    t_sorted = toda_sort(arr, mode="pure")
    assert np.allclose(t_sorted, expected, atol=0.05)

    # 3. Optimal transport soft continuous mode
    ot_sorted = optimal_transport_sort(arr, mode="pure")
    # Soft transport values should be non-decreasing
    assert np.all(np.diff(ot_sorted) >= -0.5)

    # 4. Box-Ball pure integer mode
    int_arr = [4, 1, 5, 2, 3]
    bbs_sorted = box_ball_sort(int_arr, mode="pure")
    assert np.array_equal(bbs_sorted, [1, 2, 3, 4, 5])


def test_robust_mode_handles_extremes():
    """Verify that robust mode handles extreme dynamic range inputs."""
    extreme_arr = [-1e6, 1e-6, -1e-6, 1e6]
    expected = sorted(extreme_arr)

    b_res = brockett_sort(extreme_arr, mode="robust")
    assert np.allclose(b_res, expected, rtol=1e-2, atol=1e-4)

    t_res = toda_sort(extreme_arr, mode="robust")
    assert np.allclose(t_res, expected, rtol=1e-2, atol=1e-4)

    ot_res = optimal_transport_sort(extreme_arr, mode="robust")
    assert np.allclose(ot_res, expected, rtol=1e-2, atol=1e-4)

    bbs_res = box_ball_sort(extreme_arr, mode="robust")
    assert np.allclose(bbs_res, expected, rtol=1e-2, atol=1e-4)


def test_invalid_mode_raises_value_error():
    """Verify that unsupported mode strings raise ValueError."""
    arr = [3.0, 1.0, 2.0]
    for engine in [brockett_sort, toda_sort, optimal_transport_sort, box_ball_sort]:
        with pytest.raises(ValueError, match="mode must be 'pure' or 'robust'"):
            engine(arr, mode="unsupported_mode")
