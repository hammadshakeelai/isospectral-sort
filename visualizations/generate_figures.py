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
from isospectralsort.optimal_transport import optimal_transport_sort


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


def generate_sinkhorn_figures(output_path: str = "visualizations/sinkhorn_transport.png"):
    """
    Generate 4-panel publication figure illustrating Cuturi (2013) entropic Sinkhorn
    transport and Blondel et al. (2020) differentiable permutahedron sorting.
    """
    vals = np.array([42.0, -15.0, 88.0, 0.0, 25.0, -50.0])
    n = len(vals)
    sorted_vals, diag = optimal_transport_sort(vals, epsilon=0.03, return_diagnostics=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Entropic Optimal Transport & Differentiable Sorting\n(Cuturi NeurIPS 2013 · Blondel et al. ICML 2020 · Brenier 1991)", 
                 fontsize=15, fontweight="bold", y=0.98)
    
    # 1. Quadratic Ground Cost Matrix C_ij = (x_i - rank_j)^2
    ranks = np.arange(1, n + 1, dtype=float)
    C = (vals[:, None] - ranks[None, :]) ** 2
    im1 = axes[0, 0].imshow(C, cmap="viridis", aspect="auto")
    axes[0, 0].set_title(r"1. Quadratic Cost Matrix $C_{ij} = (x_i - j)^2$", fontweight="bold")
    axes[0, 0].set_xlabel("Target Rank Index $j$")
    axes[0, 0].set_ylabel(r"Input Value Index $i$ ($x_i$)")
    axes[0, 0].set_xticks(range(n))
    axes[0, 0].set_xticklabels([f"r={j+1}" for j in range(n)])
    axes[0, 0].set_yticks(range(n))
    axes[0, 0].set_yticklabels([f"{v:.0f}" for v in vals])
    fig.colorbar(im1, ax=axes[0, 0], label="Transport Cost")
    
    # 2. Cuturi Doubly Stochastic Coupling Matrix P_eps
    P = diag.permutation_matrix
    im2 = axes[0, 1].imshow(P, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[0, 1].set_title(r"2. Cuturi Coupling Matrix $P_\varepsilon \in \mathcal{B}_n$ ($\varepsilon=0.03$)", fontweight="bold")
    axes[0, 1].set_xlabel("Target Rank Index $j$")
    axes[0, 1].set_ylabel(r"Input Value Index $i$ ($x_i$)")
    axes[0, 1].set_xticks(range(n))
    axes[0, 1].set_xticklabels([f"r={j+1}" for j in range(n)])
    axes[0, 1].set_yticks(range(n))
    axes[0, 1].set_yticklabels([f"{v:.0f}" for v in vals])
    fig.colorbar(im2, ax=axes[0, 1], label=r"Coupling Probability $P_{ij}$")
    
    # 3. Shannon Entropy Decay vs Regularization Temperature (Annealing)
    eps_range = np.logspace(-2.0, 1.2, 40)
    entropies = []
    for eps in eps_range:
        _, d = optimal_transport_sort(vals, epsilon=eps, return_diagnostics=True)
        entropies.append(d.entropy)
    axes[1, 0].semilogx(eps_range, entropies, color="darkviolet", linewidth=2.5)
    axes[1, 0].axvline(0.03, color="crimson", linestyle="--", alpha=0.7, label=r"Operating $\varepsilon=0.03$")
    axes[1, 0].set_title(r"3. Shannon Entropy $H(P) = -\sum P_{ij} \ln P_{ij}$ vs $\varepsilon$", fontweight="bold")
    axes[1, 0].set_xlabel(r"Entropic Regularization Temperature $\varepsilon$ (Log Scale)")
    axes[1, 0].set_ylabel("Shannon Entropy $H(P)$")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Blondel Differentiable Sorting Paths: s(eps) = P_eps^T @ x
    soft_paths = []
    for eps in eps_range:
        s, _ = optimal_transport_sort(vals, epsilon=eps, soft=True, return_diagnostics=True)
        soft_paths.append(s)
    soft_paths = np.array(soft_paths)
    for k in range(n):
        axes[1, 1].semilogx(eps_range, soft_paths[:, k], linewidth=2, label=f"Rank {k+1}")
    axes[1, 1].set_title(r"4. Blondel Differentiable Permutahedron Paths $s = P_\varepsilon^T x$", fontweight="bold")
    axes[1, 1].set_xlabel(r"Temperature $\varepsilon$ (High Entropy $\to$ Discrete Limit)")
    axes[1, 1].set_ylabel("Continuous Soft-Sorted Coordinates")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend(loc="upper left", fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved Sinkhorn analysis figure to: {output_path}")


if __name__ == "__main__":
    generate_brockett_figures()
    generate_soliton_spacetime()
    generate_sinkhorn_figures()
