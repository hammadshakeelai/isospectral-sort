"""
Toda Lattice Lax Pair Scattering Flow.

Implements Jürgen Moser's 1975 integrable Hamiltonian dynamical system:
    dL/dt = [B, L]
where L is a real symmetric tridiagonal Jacobi matrix:
    L = [ b_1  a_1   0   ... ]
        [ a_1  b_2  a_2  ... ]
        [  0   a_2  b_3  ... ]
and B is the skew-symmetric projection:
    B = [  0   a_1   0   ... ]
        [-a_1   0   a_2  ... ]
        [  0  -a_2   0   ... ]

As t -> +inf, off-diagonal entries a_k(t) -> 0 exponentially fast, and the diagonal
elements b_k(t) converge to the eigenvalues of L(0) in strictly descending order.
As t -> -inf, b_k(t) converge to the eigenvalues in strictly ascending order.

Reference:
    Moser, J. (1975). "Finitely many points on the line under the influence
    of an exponential potential - An integrable system."
    Lecture Notes in Physics, Vol. 38, pp. 467-497.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class TodaDiagnostics:
    """Diagnostic tracking for the Toda lattice Hamiltonian flow."""
    iterations: int
    converged: bool
    final_offdiag_norm: float
    offdiag_history: List[float]
    diagonal_history: List[np.ndarray]
    eigenvalue_drift: float


def _embed_in_jacobi_matrix(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a symmetric tridiagonal Jacobi matrix L(0) with positive subdiagonal
    whose eigenvalues are exactly equal to `values`.
    
    Uses Lanczos tridiagonalization applied to diag(values) with the uniform
    initial vector v0 = (1/sqrt(n)) * [1, ..., 1]^T.
    """
    n = len(values)
    v = np.ones(n, dtype=float) / np.sqrt(n)
    alpha = np.zeros(n, dtype=float)
    beta = np.zeros(n - 1, dtype=float)
    
    V = np.zeros((n, n), dtype=float)
    V[:, 0] = v
    
    u = values * v
    alpha[0] = float(v @ u)
    u = u - alpha[0] * v
    
    for j in range(1, n):
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-14:
            # Handle duplicate or degenerate eigenvalues
            v_cand = np.random.randn(n)
            v_cand -= V[:, :j] @ (V[:, :j].T @ v_cand)
            norm_u = np.linalg.norm(v_cand)
            v = v_cand / max(norm_u, 1e-12)
            norm_u = 1e-6
        else:
            v = u / norm_u
            
        beta[j - 1] = norm_u
        V[:, j] = v
        
        u = values * v - beta[j - 1] * V[:, j - 1]
        alpha[j] = float(v @ u)
        u = u - alpha[j] * v
        
    L0 = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    return L0, alpha, beta


def toda_sort(
    values: Union[List[float], np.ndarray],
    reverse: bool = False,
    dt: Optional[float] = None,
    max_steps: int = 10000,
    tol: float = 1e-4,
    return_diagnostics: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, TodaDiagnostics]]:
    """
    Sort an array of real numbers using the non-periodic Toda lattice Lax flow.
    
    Parameters
    ----------
    values : array-like of shape (n,)
        The numbers to be sorted.
    reverse : bool, default=False
        If True, sort in descending order (natural Moser t -> +inf limit).
        If False, sort in ascending order (Moser t -> -inf limit).
    dt : float, optional
        Integration step size. If None, an adaptive step size is chosen.
    max_steps : int, default=10000
        Maximum integration steps.
    tol : float, default=1e-4
        Tolerance for the off-diagonal norm of L(t).
    return_diagnostics : bool, default=False
        Whether to return diagnostic history.
        
    Returns
    -------
    sorted_values : np.ndarray
        The sorted array.
    diagnostics : TodaDiagnostics (optional)
        Convergence and trajectory metrics.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    
    if n <= 1:
        if return_diagnostics:
            diag = TodaDiagnostics(
                iterations=0,
                converged=True,
                final_offdiag_norm=0.0,
                offdiag_history=[0.0],
                diagonal_history=[vals.copy()],
                eigenvalue_drift=0.0
            )
            return vals.copy(), diag
        return vals.copy()

    # Embed values into Jacobi tridiagonal matrix L(0)
    L, alpha, beta = _embed_in_jacobi_matrix(vals)
    orig_sorted_evals = np.sort(vals)
    
    # Scale-adaptive step size
    val_spread = float(np.ptp(vals))
    val_norm = float(np.linalg.norm(vals))
    if dt is None:
        scale = max(1.0, val_norm)
        step_dt = 0.2 / scale
    else:
        step_dt = dt

    I = np.eye(n)
    offdiag_history = []
    diagonal_history = []
    converged = False
    step = 0
    
    # Direction sign: reverse=True is t -> +inf (descending), reverse=False is t -> -inf (ascending)
    dir_sign = 1.0 if reverse else -1.0
    
    for step in range(1, max_steps + 1):
        # Construct skew-symmetric Lax companion matrix B:
        # B_i,i+1 = dir_sign * L_i,i+1,  B_i+1,i = -dir_sign * L_i+1,i
        B = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            sub = dir_sign * L[i, i + 1]
            B[i, i + 1] = sub
            B[i + 1, i] = -sub
            
        # Exact Lie group update via Cayley transform
        A = 0.5 * step_dt * B
        U = np.linalg.solve(I - A, I + A)
        L = U @ L @ U.T
        L = 0.5 * (L + L.T)
        
        # Off-diagonal Frobenius norm
        diff_sq = np.sum(L**2) - np.sum(np.diag(L)**2)
        offdiag_norm = np.sqrt(max(0.0, diff_sq))
        
        if return_diagnostics:
            offdiag_history.append(float(offdiag_norm))
            diagonal_history.append(np.diag(L).copy())
            
        rel_tol = tol * max(1.0, val_norm)
        if offdiag_norm < rel_tol:
            converged = True
            break

    sorted_result = np.diag(L).copy()
    
    if return_diagnostics:
        current_evals = np.sort(np.linalg.eigvalsh(L))
        drift = float(np.max(np.abs(current_evals - orig_sorted_evals)))
        
        diag = TodaDiagnostics(
            iterations=step,
            converged=converged,
            final_offdiag_norm=float(offdiag_norm),
            offdiag_history=offdiag_history,
            diagonal_history=diagonal_history,
            eigenvalue_drift=drift
        )
        return sorted_result, diag
        
    return sorted_result
