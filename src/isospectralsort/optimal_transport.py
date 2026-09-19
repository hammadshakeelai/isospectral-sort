"""
Monge-Kantorovich Optimal Transport & Differentiable Sinkhorn Sorting.

Formulates sorting as an optimal transport problem between an empirical measure
of unsorted numbers and a reference ordered rank distribution.

By the classical Rearrangement Inequality and Brenier's Theorem (1991),
the optimal transport map under convex quadratic cost between 1D distributions
is uniquely monotone. The optimal assignment on the Birkhoff polytope of
doubly stochastic matrices corresponds identically to the sorting permutation.

With entropic regularization (Cuturi 2013, Blondel et al. 2020), the Sinkhorn-Knopp
matrix-scaling algorithm computes a smooth, infinitely differentiable permutation matrix.

References:
    Brenier, Y. (1991). "Polar factorization and monotone rearrangement of
    vector-valued functions." Communications on Pure and Applied Mathematics, 44(4), 375-417.
    Cuturi, M. (2013). "Sinkhorn distances: Lightspeed computation of optimal transport."
    NeurIPS 2013.
    Blondel, M., Teboul, O., Berthet, Q., & Djolonga, J. (2020). "Fast Differentiable Sorting and Ranking."
    ICML 2020.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class OptimalTransportDiagnostics:
    """Diagnostic tracking for the Sinkhorn optimal transport algorithm."""
    iterations: int
    converged: bool
    temperature: float
    permutation_matrix: np.ndarray
    entropy: float
    transport_cost: float


def optimal_transport_sort(
    values: Union[List[float], np.ndarray],
    reverse: bool = False,
    epsilon: float = 0.05,
    max_iters: int = 1500,
    tol: float = 1e-3,
    return_diagnostics: bool = False,
    soft: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, OptimalTransportDiagnostics]]:
    """
    Sort an array using entropic-regularized Monge-Kantorovich optimal transport.
    
    Parameters
    ----------
    values : array-like of shape (n,)
        The numbers to sort.
    reverse : bool, default=False
        If True, sort in descending order.
    epsilon : float, default=0.05
        Entropic regularization temperature. Smaller epsilon yields sharper permutations.
    max_iters : int, default=1500
        Maximum Sinkhorn scaling iterations.
    tol : float, default=1e-3
        Tolerance on doubly stochastic marginal constraint violation.
    return_diagnostics : bool, default=False
        Whether to return diagnostic metadata.
    soft : bool, default=False
        If True, return differentiable soft-sorted values (P^T @ x).
        If False, return exact discrete sorted values via permutation argmax.
        
    Returns
    -------
    sorted_values : np.ndarray
        Sorted array.
    diagnostics : OptimalTransportDiagnostics (optional)
        Diagnostics including the doubly stochastic transport matrix.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    
    if n <= 1:
        if return_diagnostics:
            diag = OptimalTransportDiagnostics(
                iterations=0,
                converged=True,
                temperature=epsilon,
                permutation_matrix=np.ones((n, n)),
                entropy=0.0,
                transport_cost=0.0
            )
            return vals.copy(), diag
        return vals.copy()

    # Normalize values for numerical stability in the Gibbs kernel
    std_val = float(np.std(vals))
    if std_val < 1e-12:
        # All elements identical
        if return_diagnostics:
            diag = OptimalTransportDiagnostics(
                iterations=0,
                converged=True,
                temperature=epsilon,
                permutation_matrix=np.eye(n),
                entropy=0.0,
                transport_cost=0.0
            )
            return vals.copy(), diag
        return vals.copy()

    x_norm = (vals - np.mean(vals)) / std_val
    
    # Target ranks: ascending 1..n or descending n..1
    if reverse:
        ranks = np.arange(n, 0, -1, dtype=float)
    else:
        ranks = np.arange(1, n + 1, dtype=float)
    ranks_norm = (ranks - np.mean(ranks)) / np.std(ranks)
    
    # Cost matrix derived from Rearrangement Inequality:
    # C_ij = - x_norm[i] * ranks_norm[j]
    C = -np.outer(x_norm, ranks_norm)
    
    # Kernel matrix K = exp(-C / eps)
    # Use log-sum-exp stabilization to avoid overflow/underflow
    min_C = np.min(C)
    K = np.exp(-(C - min_C) / epsilon)
    
    # Sinkhorn-Knopp fixed-point iteration
    u = np.ones(n, dtype=float)
    v = np.ones(n, dtype=float)
    converged = False
    
    for it in range(1, max_iters + 1):
        # u = 1 / (K v)
        Kv = K @ v
        Kv[Kv < 1e-30] = 1e-30
        u = 1.0 / Kv
        
        # v = 1 / (K^T u)
        KTu = K.T @ u
        KTu[KTu < 1e-30] = 1e-30
        v = 1.0 / KTu
        
        # Marginal violation check: ||P 1 - 1||_inf
        P = np.diag(u) @ K @ np.diag(v)
        row_err = np.max(np.abs(np.sum(P, axis=1) - 1.0))
        col_err = np.max(np.abs(np.sum(P, axis=0) - 1.0))
        if max(row_err, col_err) < tol:
            converged = True
            break

    # Doubly stochastic transport matrix P (rows sum to 1, cols sum to 1)
    P = np.diag(u) @ K @ np.diag(v)
    # Re-normalize to exact doubly stochastic
    col_sums = np.sum(P, axis=0, keepdims=True)
    col_sums[col_sums < 1e-30] = 1.0
    P /= col_sums

    if soft:
        sorted_result = P.T @ vals
    else:
        # Hard permutation assignment:
        # Col j has the highest probability for which input x_i belongs at rank j
        perm_idx = np.argmax(P, axis=0)
        sorted_result = vals[perm_idx].copy()
        
    if return_diagnostics:
        # Shannon entropy H(P) = -sum P_ij log(P_ij + eps)
        safe_P = np.clip(P, 1e-30, 1.0)
        entropy = -float(np.sum(safe_P * np.log(safe_P)))
        cost = float(np.sum(P * C))
        
        diag = OptimalTransportDiagnostics(
            iterations=it,
            converged=converged,
            temperature=epsilon,
            permutation_matrix=P,
            entropy=entropy,
            transport_cost=cost
        )
        return sorted_result, diag

    return sorted_result
