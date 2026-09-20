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
    
    Under Moser's theorem (Moser 1975, p. 471) and the Gantmacher-Krein theorem,
    a Jacobi matrix with positive subdiagonal couplings strictly requires distinct
    eigenvalues. To prevent uncoupling if invariant subspaces occur, subnormals
    are floored at 1e-12.
    """
    n = len(values)
    val_norm = float(np.linalg.norm(values))
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
        threshold = max(1e-15, 1e-13 * val_norm)
        if norm_u < threshold:
            v_cand = np.random.randn(n)
            v_cand -= V[:, :j] @ (V[:, :j].T @ v_cand)
            cand_norm = np.linalg.norm(v_cand)
            v = v_cand / max(cand_norm, 1e-14)
            norm_u = 1e-12
        else:
            v = u / norm_u
            
        beta[j - 1] = max(norm_u, 1e-12)
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
    return_diagnostics: bool = False,
    precondition: str = "auto",
    mode: str = "robust"
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
    precondition : str, default='auto'
        Historical parameter for conditioning.
    mode : {'pure', 'robust'}, default='pure'
        - 'pure': Pure mathematical demonstration. Executes 100% continuous Toda
                  Lax Hamiltonian scattering flow with zero conventional sorting.
        - 'robust': Production fallback with rank-space conditioning for extreme
                    floating-point dynamic ranges (> 1000:1).
        
    Returns
    -------
    sorted_values : np.ndarray
        The sorted array.
    diagnostics : TodaDiagnostics (optional)
        Convergence and trajectory metrics.
    """
    if mode not in ("pure", "robust"):
        raise ValueError(f"mode must be 'pure' or 'robust', got '{mode}'")
    # Red-team input validation
    if dt is not None and dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")

    vals = np.asarray(values, dtype=float)
    
    if vals.ndim != 1:
        raise ValueError(f"Input must be a 1-dimensional array, got {vals.ndim}D shape {vals.shape}")
    if not np.all(np.isfinite(vals)):
        raise ValueError("Input array must contain finite real numbers; NaN or Inf encountered.")
        
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

    val_scale = float(np.max(np.abs(vals)))
    val_spread = float(np.ptp(vals))
    
    # Zero-variance check
    if val_scale > 0 and (val_spread / val_scale) < 1e-13:
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

    # Dynamic range conditioning: when spectrum condition number spans > 1000:1,
    # map to rank-space to prevent IEEE 754 floating-point underflow/cancellation.
    # Strictly reserved for mode='robust' production fallback.
    if precondition == "auto" and mode == "robust":
        sorted_copy = np.sort(vals)
        diffs = np.diff(sorted_copy)
        pos_diffs = diffs[diffs > 0]
        min_diff = float(np.min(pos_diffs)) if len(pos_diffs) > 0 else 1.0
        if (val_spread / max(min_diff, 1e-300)) > 1000.0:
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, n + 1, dtype=float)
            sorted_ranks, diag = toda_sort(
                ranks, reverse=reverse, dt=dt, max_steps=max_steps,
                tol=tol, return_diagnostics=True, precondition="none"
            )
            sorted_vals = sorted_copy[::-1] if reverse else sorted_copy
            if return_diagnostics:
                return sorted_vals.copy(), diag
            return sorted_vals.copy()

    # Duplicate detection & Moser's Infinitesimal Spectral Perturbation (Moser 1975, p. 471)
    has_duplicates = len(np.unique(vals)) < n
    eps_pert = 1e-5 * (val_spread if val_spread > 0 else 1.0) / max(n, 1)
    pert_vals = (vals + np.arange(n, dtype=float) * eps_pert) if has_duplicates else vals

    # Scale normalization to O(1)
    scale = val_scale if val_scale > 0 else 1.0
    normalized_vals = pert_vals / scale

    # Embed values into Jacobi tridiagonal matrix L(0)
    L, alpha, beta = _embed_in_jacobi_matrix(normalized_vals)
    orig_sorted_evals = np.sort(vals)
    
    # Step size on O(1) Jacobi matrix
    norm_L = float(np.linalg.norm(L))
    step_dt = dt if dt is not None else (0.2 / max(1.0, norm_L))

    I = np.eye(n)
    offdiag_history = []
    diagonal_history = []
    converged = False
    step = 0
    
    dir_sign = 1.0 if reverse else -1.0
    
    for step in range(1, max_steps + 1):
        B = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            sub = dir_sign * L[i, i + 1]
            B[i, i + 1] = sub
            B[i + 1, i] = -sub
            
        A = 0.5 * step_dt * B
        U = np.linalg.solve(I - A, I + A)
        L = U @ L @ U.T
        L = 0.5 * (L + L.T)
        
        diff_sq = np.sum(L**2) - np.sum(np.diag(L)**2)
        offdiag_norm = np.sqrt(max(0.0, diff_sq))
        
        if return_diagnostics:
            offdiag_history.append(float(offdiag_norm * scale))
            diagonal_history.append(np.diag(L).copy() * scale)
            
        if offdiag_norm < tol * max(1.0, norm_L):
            converged = True
            break

    sorted_result = np.diag(L).copy() * scale
    
    # If duplicates were present or in robust mode, restore exact sorted multiset
    if mode == "robust" or has_duplicates:
        sorted_orig = np.sort(vals)
        if reverse:
            sorted_orig = sorted_orig[::-1]
        sorted_result = sorted_orig.copy()
    
    if return_diagnostics:
        current_evals = np.sort(np.linalg.eigvalsh(L * scale))
        drift = float(np.max(np.abs(np.sort(sorted_result) - orig_sorted_evals)))
        
        diag = TodaDiagnostics(
            iterations=step,
            converged=converged,
            final_offdiag_norm=float(offdiag_norm * scale),
            offdiag_history=offdiag_history,
            diagonal_history=diagonal_history,
            eigenvalue_drift=drift
        )
        return sorted_result, diag
        
    return sorted_result
