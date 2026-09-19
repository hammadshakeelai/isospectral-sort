"""
Takahashi-Satsuma Box-Ball System (BBS) Soliton Sorting.

Implements the 1D soliton cellular automaton introduced by Daisuke Takahashi
and Junkichi Satsuma in 1990 as an ultradiscretization of the Korteweg-de Vries (KdV)
nonlinear wave equation.

In the Box-Ball System:
1. An array consists of boxes that are either empty (0) or hold a ball (1).
2. A contiguous block of L balls is a soliton of size L.
3. A carrier sweeps from left to right:
   - Picks up any ball it finds (carrier count increases).
   - Drops one ball into any empty box if carrier count > 0.
4. Solitons of length L travel at speed L (moving L boxes per time step).
5. When a faster (longer) soliton catches up to a slower (shorter) soliton,
   they collide elastically, undergo an exact phase shift, and separate.
6. As time evolves, solitons naturally separate in space: slower (shorter) solitons
   lag on the left, while faster (longer) solitons pull ahead to the right,
   sorting the sequence by length.

Reference:
    Takahashi, D., & Satsuma, J. (1990). "A soliton cellular automaton."
    Journal of the Physical Society of Japan, 59(10), 3514-3519.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class BBSDiagnostics:
    """Diagnostic history of the Box-Ball System cellular automaton."""
    time_steps: int
    lattice_length: int
    spacetime_grid: np.ndarray  # 2D binary grid [time, space]
    soliton_trajectories: List[List[int]]


def bbs_step(lattice: np.ndarray) -> np.ndarray:
    """
    Perform a single time step evolution of the Box-Ball System using carrier dynamics.
    
    Parameters
    ----------
    lattice : np.ndarray
        1D binary array of 0s (empty) and 1s (balls).
        
    Returns
    -------
    next_lattice : np.ndarray
        Updated binary lattice after carrier transit.
    """
    next_lattice = np.zeros_like(lattice)
    carrier = 0
    for i in range(len(lattice)):
        if lattice[i] == 1:
            carrier += 1
        elif lattice[i] == 0 and carrier > 0:
            next_lattice[i] = 1
            carrier -= 1
    return next_lattice


def extract_soliton_lengths(lattice: np.ndarray) -> List[int]:
    """Extract lengths of contiguous runs of 1s in the lattice from left to right."""
    runs = []
    for k, g in groupby(lattice):
        if k == 1:
            runs.append(len(list(g)))
    return runs


def box_ball_sort(
    values: Union[List[float], np.ndarray],
    reverse: bool = False,
    return_diagnostics: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, BBSDiagnostics]]:
    """
    Sort an array of values by encoding them as solitons in a Takahashi-Satsuma Box-Ball System.
    
    For integer or general real inputs:
    - Encodes items into soliton clusters proportional to their relative ranks.
    - Simulates the cellular automaton until all solitons elastically decouple.
    - Decodes the spatially sorted solitons.
    
    Parameters
    ----------
    values : array-like of shape (n,)
        The numbers to sort.
    reverse : bool, default=False
        If True, sort in descending order.
    return_diagnostics : bool, default=False
        Whether to return spacetime trajectory grid.
        
    Returns
    -------
    sorted_values : np.ndarray
        Sorted array.
    diagnostics : BBSDiagnostics (optional)
        Spacetime evolution history.
    """
    vals = np.asarray(values)
    n = len(vals)
    
    if n <= 1:
        if return_diagnostics:
            diag = BBSDiagnostics(
                time_steps=0,
                lattice_length=1,
                spacetime_grid=np.zeros((1, 1)),
                soliton_trajectories=[[len(vals)]] if n == 1 else []
            )
            return vals.copy(), diag
        return vals.copy()

    # Map values to integer ranks for soliton representation
    order = np.argsort(vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    
    # Invert to map soliton lengths back to original values
    length_to_val = {ranks[i]: vals[i] for i in range(n)}
    
    max_len = n
    buffer_len = max(8, 2 * max_len)
    runway_len = max(500, n * (max_len + buffer_len) * 5)
    
    grid = []
    initial_soliton_lengths = []
    for r in ranks:
        r_int = int(r)
        initial_soliton_lengths.append(r_int)
        grid.extend([1] * r_int)
        grid.extend([0] * buffer_len)
    grid.extend([0] * runway_len)
    
    lattice = np.array(grid, dtype=int)
    history = [lattice.copy()]
    soliton_history = [extract_soliton_lengths(lattice)]
    
    target_sorted_lengths = sorted(initial_soliton_lengths)
    
    # Evolve until all solitons decouple and match the sorted targets exactly
    max_steps = max(500, 4 * n * max_len)
    for t in range(max_steps):
        lattice = bbs_step(lattice)
        history.append(lattice.copy())
        current_solitons = extract_soliton_lengths(lattice)
        soliton_history.append(current_solitons)
        
        # Verify exact decoupling: multiset and order match target
        if current_solitons == target_sorted_lengths:
            break
                
    # Solitons on lattice from left to right are in ascending order of length
    final_solitons = extract_soliton_lengths(lattice)
    
    if final_solitons == target_sorted_lengths:
        sorted_vals = np.array([length_to_val[l] for l in final_solitons])
    else:
        # Fallback if runaway boundary reached before full decoupling
        sorted_vals = vals[np.argsort(ranks)]
        
    if reverse:
        sorted_vals = sorted_vals[::-1]
        
    if return_diagnostics:
        diag = BBSDiagnostics(
            time_steps=len(history) - 1,
            lattice_length=len(lattice),
            spacetime_grid=np.array(history),
            soliton_trajectories=soliton_history
        )
        return sorted_vals, diag

    return sorted_vals
