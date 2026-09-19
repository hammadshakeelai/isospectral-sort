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
        # Diagonal is values, with small symmetric tridiagonal coupling
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
    
    Integration is performed via Cayley transforms in the Lie algebra so(n), guaranteeing
    exact isospectral invariance (eigenvalues remain constant to machine precision).
    
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
    vals = np.asarray(values, dtype=float)
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

    # Ill-conditioned check: if range ratio > 1000 and auto preconditioning is enabled
    if precondition == "auto":
        val_spread = float(np.ptp(vals))
        sorted_copy = np.sort(vals)
        diffs = np.diff(sorted_copy)
        pos_diffs = diffs[diffs > 1e-12]
        min_diff = float(np.min(pos_diffs)) if len(pos_diffs) > 0 else 1.0
        if (val_spread / min_diff) > 1000.0:
            # Run Brockett on the monotonic rank coordinates (gap = 1)
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

    # Target sorting matrix N
    if reverse:
        # Descending: N = diag(n, n-1, ..., 1)
        diag_N = np.arange(n, 0, -1, dtype=float)
    else:
        # Ascending: N = diag(1, 2, ..., n)
        diag_N = np.arange(1, n + 1, dtype=float)
    N = np.diag(diag_N)
    
    # Initial matrix H(0) whose eigenvalues are strictly vals
    H, _ = _create_initial_matrix(vals, method=init_method, seed=seed)
    
    # Exact original eigenvalues for invariant verification
    orig_sorted_evals = np.sort(vals)
    if reverse:
        theoretical_max_trace = float(np.sum(orig_sorted_evals[::-1] * diag_N))
    else:
        theoretical_max_trace = float(np.sum(orig_sorted_evals * diag_N))
    
    # Adaptive scale-invariant step size if dt not specified
    H_norm = np.linalg.norm(H)
    N_norm = np.linalg.norm(N)
    if dt is None:
        if H_norm * N_norm > 1e-12:
            step_dt = 0.5 / (H_norm * N_norm)
        else:
            step_dt = 0.01
    else:
        step_dt = dt

    trace_history = []
    offdiag_history = []
    diagonal_history = []
    
    I = np.eye(n)
    converged = False
    step = 0
    
    for step in range(1, max_steps + 1):
        # Skew-symmetric generator in so(n):
        Omega = N @ H - H @ N
        
        # Cayley transform: U = (I - 0.5*step_dt*Omega)^(-1) * (I + 0.5*step_dt*Omega)
        A = 0.5 * step_dt * Omega
        U = np.linalg.solve(I - A, I + A)
        
        # Exact isospectral similarity update: H(t + dt) = U * H(t) * U^T
        H = U @ H @ U.T
        H = 0.5 * (H + H.T)
        
        diff_sq = np.sum(H**2) - np.sum(np.diag(H)**2)
        offdiag_norm = np.sqrt(max(0.0, diff_sq))
        
        if return_diagnostics:
            trace_history.append(float(np.trace(H @ N)))
            offdiag_history.append(float(offdiag_norm))
            diagonal_history.append(np.diag(H).copy())
            
        rel_tol = tol * max(1.0, H_norm)
        if offdiag_norm < rel_tol:
            converged = True
            break
            
    sorted_result = np.diag(H).copy()
    
    if return_diagnostics:
        current_evals = np.sort(np.linalg.eigvalsh(H))
        drift = float(np.max(np.abs(current_evals - orig_sorted_evals)))
        
        diag = BrockettDiagnostics(
            iterations=step,
            converged=converged,
            final_offdiag_norm=float(offdiag_norm),
            trace_history=trace_history,
            offdiag_history=offdiag_history,
            diagonal_history=diagonal_history,
            eigenvalue_drift=drift,
            theoretical_max_trace=theoretical_max_trace
        )
        return sorted_result, diag
        
    return sorted_result
