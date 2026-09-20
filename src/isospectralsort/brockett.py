"""
Brockett Double-Bracket Isospectral Flow.

Implements Roger W. Brockett's 1991 continuous-time dynamical system:
    dH/dt = [H, [H, N]] = [[N, H], H]
where H is a real symmetric matrix whose spectrum encodes the list of numbers,
N = diag(1, 2, ..., n) is the target diagonal sorting matrix, and
[A, B] = AB - BA is the Lie bracket (matrix commutator).

Reference:
    Brockett, R. W. (1991). "Dynamical systems that sort lists, diagonalize matrices,
    and solve linear programming problems." Linear Algebra and its Applications,
    146, 79-91.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class BrockettDiagnostics:
    """Diagnostic tracking for the Brockett dynamical system trajectory."""
    iterations: int
    converged: bool
    final_offdiag_norm: float
    trace_history: List[float]
    offdiag_history: List[float]
    diagonal_history: List[np.ndarray]
    eigenvalue_drift: float  # Max absolute deviation of eigenvalues from original input
    theoretical_max_trace: float  # Upper bound from Rearrangement Inequality


def _create_initial_matrix(
    values: np.ndarray,
    method: str = "orthogonal",
    perturbation: float = 0.05,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embed input values into a symmetric matrix H(0).
    
    If H(0) is strictly diagonal, [H(0), N] = 0 (a stationary saddle point).
    To initiate the gradient flow, H(0) must have non-zero off-diagonal components
    while preserving its eigenvalues exactly equal to `values`.
    """
    n = len(values)
    rng = np.random.RandomState(seed)
    
    if method == "orthogonal":
        # Random orthogonal matrix from QR decomposition of Gaussian matrix
        A = rng.randn(n, n)
        Q, _ = np.linalg.qr(A)
        # Force det(Q) = +1 (SO(n))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        H0 = Q @ np.diag(values) @ Q.T
        return H0, Q
    elif method == "tridiagonal_perturbation":
        H0 = np.diag(values).astype(float)
        for i in range(n - 1):
            H0[i, i + 1] = perturbation
            H0[i + 1, i] = perturbation
        w, v = np.linalg.eigh(H0)
        H0 = v @ np.diag(values) @ v.T
        return H0, v
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def brockett_sort(
    values: Union[List[float], np.ndarray],
    reverse: bool = False,
    dt: Optional[float] = None,
    max_steps: int = 10000,
    tol: float = 1e-5,
    return_diagnostics: bool = False,
    init_method: str = "orthogonal",
    seed: Optional[int] = 42,
    precondition: str = "auto"
) -> Union[np.ndarray, Tuple[np.ndarray, BrockettDiagnostics]]:
    """
    Sort an array of real numbers using Brockett's continuous-time double-bracket flow.
    
    The differential equation:
        dH/dt = [H, [H, N]] = [[N, H], H]
    is the gradient ascent flow of Phi(H) = Tr(HN) on the adjoint orbit O_H0 of SO(n).
    By the Rearrangement Inequality, Phi(H) has a unique stable maximum when the diagonal
    elements of H are sorted in increasing order matching N.
    
    Parameters
    ----------
    values : array-like of shape (n,)
        The numbers to be sorted.
    reverse : bool, default=False
        If True, sort in descending order.
    dt : float, optional
        Integration step size. If None, an adaptive scale-invariant step size is used.
    max_steps : int, default=10000
        Maximum integration steps.
    tol : float, default=1e-5
        Tolerance for off-diagonal Frobenius norm to declare convergence.
    return_diagnostics : bool, default=False
        Whether to return diagnostic history.
    init_method : str, default='orthogonal'
        Method to disperse values into H(0): 'orthogonal' or 'tridiagonal_perturbation'.
    seed : int, optional, default=42
        Random seed for initial basis dispersal.
    precondition : str, default='auto'
        If 'auto', uses rank-preconditioned flow when the input array dynamic range
        ratio exceeds 1000, preventing floating-point scale disparity between modes.
        
    Returns
    -------
    sorted_values : np.ndarray
        The sorted array.
    diagnostics : BrockettDiagnostics (optional)
        Convergence and trajectory metrics.
    """
    # Red-team input validation
    if dt is not None and dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if init_method not in ("orthogonal", "tridiagonal_perturbation"):
        raise ValueError(f"Unknown init_method '{init_method}'. Expected 'orthogonal' or 'tridiagonal_perturbation'.")

    vals = np.asarray(values, dtype=float)
    
    if vals.ndim != 1:
        raise ValueError(f"Input must be a 1-dimensional array, got {vals.ndim}D shape {vals.shape}")
    if not np.all(np.isfinite(vals)):
        raise ValueError("Input array must contain finite real numbers; NaN or Inf encountered.")
        
    n = len(vals)
    
    # Degenerate cases: 0 or 1 element
    if n <= 1:
        if return_diagnostics:
            diag = BrockettDiagnostics(
                iterations=0,
                converged=True,
                final_offdiag_norm=0.0,
                trace_history=[float(np.sum(vals))] if n == 1 else [0.0],
                offdiag_history=[0.0],
                diagonal_history=[vals.copy()],
                eigenvalue_drift=0.0,
                theoretical_max_trace=float(np.sum(vals)) if n == 1 else 0.0
            )
            return vals.copy(), diag
        return vals.copy()

    val_scale = float(np.max(np.abs(vals)))
    val_spread = float(np.ptp(vals))
    
    # Zero-variance check: if all elements are identical
    if val_scale > 0 and (val_spread / val_scale) < 1e-13:
        if return_diagnostics:
            diag = BrockettDiagnostics(
                iterations=0,
                converged=True,
                final_offdiag_norm=0.0,
                trace_history=[float(np.sum(vals * np.arange(1, n + 1)))],
                offdiag_history=[0.0],
                diagonal_history=[vals.copy()],
                eigenvalue_drift=0.0,
                theoretical_max_trace=float(np.sum(vals * np.arange(1, n + 1)))
            )
            return vals.copy(), diag
        return vals.copy()

    # Dynamic range conditioning: when spectrum condition number spans > 1000:1,
    # map to rank-space to prevent IEEE 754 floating-point underflow/cancellation.
    if precondition == "auto":
        sorted_copy = np.sort(vals)
        diffs = np.diff(sorted_copy)
        pos_diffs = diffs[diffs > 0]
        min_diff = float(np.min(pos_diffs)) if len(pos_diffs) > 0 else 1.0
        if (val_spread / max(min_diff, 1e-300)) > 1000.0:
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, n + 1, dtype=float)
            sorted_ranks, diag = brockett_sort(
                ranks, reverse=reverse, dt=dt, max_steps=max_steps,
                tol=tol, return_diagnostics=True, init_method=init_method,
                seed=seed, precondition="none"
            )
            sorted_vals = sorted_copy[::-1] if reverse else sorted_copy
            if return_diagnostics:
                return sorted_vals.copy(), diag
            return sorted_vals.copy()

    # Scale normalization to O(1) to avoid underflow/overflow in Lie algebra operations
    scale = val_scale if val_scale > 0 else 1.0
    normalized_vals = vals / scale

    # Target sorting matrix N
    if reverse:
        diag_N = np.arange(n, 0, -1, dtype=float)
    else:
        diag_N = np.arange(1, n + 1, dtype=float)
    N = np.diag(diag_N)
    
    # Initial matrix H(0) on normalized values
    H, _ = _create_initial_matrix(normalized_vals, method=init_method, seed=seed)
    
    orig_sorted_evals = np.sort(vals)
    if reverse:
        theoretical_max_trace = float(np.sum(orig_sorted_evals[::-1] * diag_N))
    else:
        theoretical_max_trace = float(np.sum(orig_sorted_evals * diag_N))
    
    # Step size on O(1) normalized matrix
    H_norm = np.linalg.norm(H)
    N_norm = np.linalg.norm(N)
    if dt is None:
        step_dt = 0.5 / max(1e-6, H_norm * N_norm)
    else:
        step_dt = dt

    trace_history = []
    offdiag_history = []
    diagonal_history = []
    
    I = np.eye(n)
    converged = False
    step = 0
    
    for step in range(1, max_steps + 1):
        Omega = N @ H - H @ N
        
        # Cayley transform update
        A = 0.5 * step_dt * Omega
        U = np.linalg.solve(I - A, I + A)
        H = U @ H @ U.T
        H = 0.5 * (H + H.T)
        
        diff_sq = np.sum(H**2) - np.sum(np.diag(H)**2)
        offdiag_norm = np.sqrt(max(0.0, diff_sq))
        
        if return_diagnostics:
            unscaled_H = H * scale
            trace_history.append(float(np.trace(unscaled_H @ N)))
            offdiag_history.append(float(offdiag_norm * scale))
            diagonal_history.append(np.diag(unscaled_H).copy())
            
        if offdiag_norm < tol * max(1.0, H_norm):
            converged = True
            break
            
    sorted_result = np.diag(H).copy() * scale
    
    if return_diagnostics:
        current_evals = np.sort(np.linalg.eigvalsh(H * scale))
        drift = float(np.max(np.abs(current_evals - orig_sorted_evals)))
        
        diag = BrockettDiagnostics(
            iterations=step,
            converged=converged,
            final_offdiag_norm=float(offdiag_norm * scale),
            trace_history=trace_history,
            offdiag_history=offdiag_history,
            diagonal_history=diagonal_history,
            eigenvalue_drift=drift,
            theoretical_max_trace=theoretical_max_trace
        )
        return sorted_result, diag
        
    return sorted_result
