"""
Publication-Quality Figure Generator for IsospectralSort.

Generates 4-panel mathematical convergence analysis figures and
soliton spacetime diagrams.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.brockett import brockett_sort
from isospectralsort.box_ball import box_ball_sort


def generate_brockett_figures(output_path: str = "visualizations/brockett_convergence.png"):
    """Generate 4-panel figure illustrating Brockett double-bracket flow dynamics."""
    vals = [42.0, -15.0, 88.0, 0.0, 25.0, -50.0]
    sorted_vals, diag = brockett_sort(vals, return_diagnostics=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Roger Brockett's Double-Bracket Flow: $\\dot{H} = [H, [H, N]]$", fontsize=16, fontweight="bold", y=0.98)
    
    # 1. Diagonal Trajectories (Eigenvalue alignment)
    diag_history = np.array(diag.diagonal_history)
    steps = np.arange(len(diag_history))
    ax1 = axes[0, 0]
    for i in range(diag_history.shape[1]):
        ax1.plot(steps, diag_history[:, i], linewidth=2, label=f"$H_{{{i+1},{i+1}}}(t)$")
    ax1.set_title("1. Continuous Diagonal Trajectories (Sorting Dynamics)", fontweight="bold")
    ax1.set_xlabel("Integration Steps")
    ax1.set_ylabel("Diagonal Entry Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="right", fontsize=9)
    
    # 2. Off-Diagonal Frobenius Norm Decay (Log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(steps, diag.offdiag_history, color="crimson", linewidth=2.2)
    ax2.set_title("2. Off-Diagonal Frobenius Norm Decay (Decoupling)", fontweight="bold")
    ax2.set_xlabel("Integration Steps")
    ax2.set_ylabel(r"$\|H - \mathrm{diag}(H)\|_F$ (Log Scale)")
    ax2.grid(True, which="both", alpha=0.3)
    
    # 3. Lyapunov Potential: Trace Functional Ascent
    ax3 = axes[1, 0]
    ax3.plot(steps, diag.trace_history, color="forestgreen", linewidth=2.2, label=r"$\Phi(H) = \mathrm{Tr}(HN)$")
    ax3.axhline(diag.theoretical_max_trace, color="black", linestyle="--", alpha=0.8,
                label=f"Theoretical Upper Bound ({diag.theoretical_max_trace:.1f})")
    ax3.set_title("3. Lyapunov Function Ascent (Rearrangement Inequality)", fontweight="bold")
    ax3.set_xlabel("Integration Steps")
    ax3.set_ylabel("Trace Potential")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")
    
    # 4. Final Sorted Spectrum vs Theoretical Sorting
    ax4 = axes[1, 1]
    expected = sorted(vals)
    indices = np.arange(1, len(vals) + 1)
    bar_width = 0.35
    ax4.bar(indices - bar_width/2, expected, bar_width, label="Classical Sort", color="steelblue", alpha=0.8)
    ax4.bar(indices + bar_width/2, sorted_vals, bar_width, label="Brockett Asymptotic Diagonal", color="darkorange", alpha=0.8)
    ax4.set_title("4. Classical Sort vs. Flow Equilibrium Spectrum", fontweight="bold")
    ax4.set_xlabel("Rank Position $i$")
    ax4.set_ylabel("Value")
    ax4.set_xticks(indices)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved Brockett analysis figure to: {output_path}")


def generate_soliton_spacetime(output_path: str = "visualizations/soliton_spacetime.png"):
    """Generate spacetime 2D grid diagram of the Takahashi-Satsuma Box-Ball System."""
    vals = [4, 1, 3, 2]
    _, diag = box_ball_sort(vals, return_diagnostics=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    grid = diag.spacetime_grid
    
    # Crop horizontal space for aesthetic clarity
    active_cols = np.where(grid.any(axis=0))[0]
    max_col = min(grid.shape[1], active_cols[-1] + 10) if len(active_cols) > 0 else grid.shape[1]
    cropped = grid[:, :max_col]
    
    ax.imshow(cropped, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_title("Takahashi-Satsuma Box-Ball Soliton Spacetime Evolution (Ultradiscrete KdV)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lattice Coordinate $x$ (Boxes)")
    ax.set_ylabel("Discrete Time $t$ (Carrier Sweeps)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved Soliton Spacetime figure to: {output_path}")


if __name__ == "__main__":
    generate_brockett_figures()
    generate_soliton_spacetime()
