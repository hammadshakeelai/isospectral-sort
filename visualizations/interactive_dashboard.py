"""
Interactive Streamlit Dashboard & Paper RAG Explorer for IsospectralSort.

Run locally:
    streamlit run visualizations/interactive_dashboard.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.brockett import brockett_sort
from isospectralsort.toda import toda_sort
from isospectralsort.optimal_transport import optimal_transport_sort
from isospectralsort.box_ball import box_ball_sort
from isospectralsort.rag.engine import PaperRAG

import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="IsospectralSort Explorer",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 IsospectralSort: Continuous & Geometric Sorting in Python")
st.markdown("""
Traditional computer science sorts via discrete comparisons (`if a < b then swap`).
**IsospectralSort** solves sorting through **Lie algebra gradient flows**, **Hamiltonian mechanics**, 
**ultradiscrete solitons**, and **optimal transport**.
""")

tabs = st.tabs(["🚀 Live Algorithm Simulator", "📚 Research Papers & RAG Explorer", "📐 Mathematical Foundations"])

with tabs[0]:
    st.subheader("Interactive Dynamical Sorting Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        algo = st.selectbox(
            "Select Mathematical Engine",
            ["Brockett Double-Bracket Flow", "Toda Lattice Lax Flow", "Optimal Transport (Sinkhorn)", "Box-Ball System (BBS)"]
        )
        
        default_input = "42.0, -15.0, 88.0, 0.0, 25.0, -50.0"
        user_input = st.text_input("Input Numbers (comma-separated)", default_input)
        reverse = st.checkbox("Descending Order", False)
        
        try:
            numbers = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        except Exception:
            st.error("Please enter valid comma-separated numbers.")
            numbers = [42.0, -15.0, 88.0, 0.0, 25.0, -50.0]

    with col2:
        if algo == "Brockett Double-Bracket Flow":
            sorted_vals, diag = brockett_sort(numbers, reverse=reverse, return_diagnostics=True)
            
            st.success(f"**Sorted Output:** `{[round(x, 2) for x in sorted_vals]}`")
            st.metric("Converged", str(diag.converged), f"Eigenvalue Drift: {diag.eigenvalue_drift:.2e}")
            st.metric("Iterations", diag.iterations, f"Final Off-Diagonal Norm: {diag.final_offdiag_norm:.2e}")
            
            # Plot diagonal trajectories
            fig, ax = plt.subplots(figsize=(8, 4))
            diag_arr = np.array(diag.diagonal_history)
            for i in range(diag_arr.shape[1]):
                ax.plot(diag_arr[:, i], label=f"$H_{{{i+1},{i+1}}}(t)$")
            ax.set_title("Continuous Diagonal Trajectories Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        elif algo == "Toda Lattice Lax Flow":
            sorted_vals, diag = toda_sort(numbers, reverse=reverse, return_diagnostics=True)
            st.success(f"**Sorted Output:** `{[round(x, 2) for x in sorted_vals]}`")
            st.metric("Converged", str(diag.converged), f"Eigenvalue Drift: {diag.eigenvalue_drift:.2e}")
            
        elif algo == "Optimal Transport (Sinkhorn)":
            sorted_vals, diag = optimal_transport_sort(numbers, reverse=reverse, return_diagnostics=True)
            st.success(f"**Sorted Output:** `{[round(x, 2) for x in sorted_vals]}`")
            st.metric("Entropy", f"{diag.entropy:.4f}", f"Transport Cost: {diag.transport_cost:.4f}")
            
            # Display transport matrix heatmap
            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.matshow(diag.permutation_matrix, cmap="Blues")
            fig.colorbar(cax)
            ax.set_title("Doubly Stochastic Permutation Matrix $P$", pad=15)
            st.pyplot(fig)
            plt.close()

        elif algo == "Box-Ball System (BBS)":
            sorted_vals, diag = box_ball_sort(numbers, reverse=reverse, return_diagnostics=True)
            st.success(f"**Sorted Output:** `{[round(x, 2) for x in sorted_vals]}`")
            st.metric("Soliton Time Steps", diag.time_steps, f"Lattice Length: {diag.lattice_length}")


with tabs[1]:
    st.subheader("Semantic Paper Knowledge Base & RAG Query")
    st.markdown("Ask technical questions directly against the indexed research papers:")
    
    rag = PaperRAG()
    query = st.text_input("Enter your research question:", "How does Brockett's flow preserve eigenvalues?")
    
    if st.button("Search Papers") and query:
        results = rag.query(query, top_k=2)
        for rank, (chunk, score) in enumerate(results, 1):
            st.markdown(f"### [{rank}] {chunk.paper_title} (Score: `{score:.4f}`)")
            st.caption(f"**Author:** {chunk.author} | **Section:** {chunk.section_title}")
            st.markdown(chunk.content)
            if chunk.equations:
                st.markdown("**Key Equations:**")
                for eq in chunk.equations[:2]:
                    st.latex(eq)
            st.divider()


with tabs[2]:
    st.subheader("Mathematical Comparison: 4 Paradigms of Pure-Math Sorting")
    st.markdown(r"""
    | Paradigm | Mathematical Space | Governing Equation | Physical / Geometric Interpretation |
    | :--- | :--- | :--- | :--- |
    | **Brockett Flow** | Lie algebra $\mathfrak{so}(n)$, Symplectic Flag Manifold | $\dot{H} = [H, [H, N]]$ | Gradient ascent on adjoint orbit maximizing trace $\mathrm{Tr}(HN)$. |
    | **Toda Lattice** | Tridiagonal Jacobi Matrices, Phase Space | $\dot{L} = [B, L]$ | Scattering of $n$ particles under exponential repulsive spring forces. |
    | **Optimal Transport** | Birkhoff Polytope $\mathcal{U}(a, b)$, Wasserstein Metric | $\min_{P} \langle P, C \rangle - \varepsilon H(P)$ | Monge-Kantorovich monotone rearrangement via Sinkhorn scaling. |
    | **Box-Ball System** | Tropical Semiring $(\mathbb{R}, \max, +)$ | Carrier cellular automaton | Solitons colliding elastically with speeds $v = L$ in discrete spacetime. |
    """)
