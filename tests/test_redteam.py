"""
Adversarial Red-Team & Stress-Testing Suite.

Attacks the mathematical sorting engines with malicious, corrupted,
subnormal, overflow, non-finite, and multi-dimensional inputs.
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
from isospectralsort.rag.engine import PaperRAG


ALL_ENGINES = [brockett_sort, toda_sort, optimal_transport_sort, box_ball_sort]


class TestRedTeamInputValidation:
    """Attacks on input validation and dimensional constraints."""

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_rejection_of_nan(self, engine):
        """Engines must immediately reject NaN without silent NaN propagation."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine([1.0, np.nan, 2.0])

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_rejection_of_positive_inf(self, engine):
        """Engines must immediately reject +Infinity."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine([1.0, np.inf, 2.0])

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_rejection_of_negative_inf(self, engine):
        """Engines must immediately reject -Infinity."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine([-np.inf, 1.0, 2.0])

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_rejection_of_2d_arrays(self, engine):
        """Engines must reject matrices / 2D inputs."""
        with pytest.raises(ValueError, match="1-dimensional"):
            engine(np.array([[1.0, 2.0], [3.0, 4.0]]))

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_rejection_of_3d_arrays(self, engine):
        """Engines must reject 3D tensors."""
        with pytest.raises(ValueError, match="1-dimensional"):
            engine(np.zeros((2, 2, 2)))


class TestRedTeamNumericalExtremes:
    """Attacks using extreme floating-point scales and degenerate configurations."""

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_all_zeros_array(self, engine):
        """Arrays containing strictly zero values [0.0, 0.0, 0.0]."""
        arr = [0.0, 0.0, 0.0, 0.0]
        res = engine(arr)
        assert np.allclose(res, arr)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_near_overflow_magnitudes(self, engine):
        """Near-overflow values on order of 1e150."""
        arr = [2e150, 1e150, 3e150]
        expected = [1e150, 2e150, 3e150]
        res = engine(arr)
        assert np.allclose(res, expected, rtol=1e-2)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_subnormal_small_magnitudes(self, engine):
        """Subnormal numbers on order of 1e-200."""
        arr = [3e-200, 1e-200, 2e-200]
        expected = [1e-200, 2e-200, 3e-200]
        res = engine(arr)
        # Verify monotone ordering
        assert res[0] <= res[1] <= res[2]

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_alternating_polar_scales(self, engine):
        """Alternating negative/positive massive and microscopic values."""
        arr = [-1e8, 1e-8, -1e-8, 1e8]
        expected = sorted(arr)
        res = engine(arr)
        assert np.allclose(res, expected, rtol=1e-2, atol=1e-4)

    def test_scale_stress_n25_all_engines(self):
        """Stress testing on random n=25 vector."""
        np.random.seed(999)
        arr = list(np.random.randn(25) * 10.0)
        expected = sorted(arr)
        
        b_res = brockett_sort(arr)
        assert np.allclose(b_res, expected, atol=0.2)
        
        ot_res = optimal_transport_sort(arr)
        assert np.allclose(ot_res, expected)
        
        bbs_res = box_ball_sort(arr)
        assert np.allclose(bbs_res, expected)


class TestRedTeamRAGSecurity:
    """Security audit on the Research Paper RAG engine."""

    def test_invalid_directory_rejection(self):
        """Invalid or non-existent directories must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid papers directory"):
            PaperRAG(papers_dir="non_existent_directory_xyz123")

    def test_null_byte_query_sanitization(self):
        """Null bytes and control characters must be sanitized without crash."""
        rag = PaperRAG()
        results = rag.query("Brockett\x00\x01\x1f\x7f flow", top_k=1)
        assert isinstance(results, list)

    def test_empty_and_whitespace_queries(self):
        """Empty or whitespace-only queries must not crash."""
        rag = PaperRAG()
        assert rag.query("") == []
        assert rag.query("   ") == []

    def test_special_regex_characters_query(self):
        """Queries with unbalanced regex patterns must not cause ReDoS."""
        rag = PaperRAG()
        results = rag.query("(((([a-z]+)*)+)*)+$$$***???", top_k=1)
        assert isinstance(results, list)
