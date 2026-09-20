# Sinkhorn Distances: Lightspeed Computation of Optimal Transport

**Author:** Marco Cuturi  
**Affiliation:** Graduate School of Informatics, Kyoto University, Japan  
**Conference:** *Advances in Neural Information Processing Systems (NeurIPS 2013)*, Lake Tahoe, Nevada.

---

## Abstract

Optimal transport distances define a natural and geometrically sound framework to compare probability distributions and feature histograms. Despite their appealing mathematical properties and empirical performance, their high computational cost ($O(n^3 \log n)$ via discrete network simplex or interior point methods) has historically limited their scalability in large-scale machine learning. 

We propose an entropic-regularized formulation of the optimal transport problem. By penalizing the transport coupling with Shannon entropy, the resulting strictly convex optimization problem can be solved at lightspeed using the classical Sinkhorn-Knopp matrix-scaling algorithm. The algorithm executes entirely through fast matrix-vector products and GPU-parallelizable operations, computing an infinitely differentiable approximation of the optimal transport distance and monotone assignment permutation.

---

## 1. Entropic Regularization of Optimal Transport

Given two discrete probability distributions $r, c \in \Sigma_n$ on the simplex and an $n \times n$ ground cost matrix $M$, the standard Kantorovich problem is:

$$d_M(r, c) = \min_{P \in U(r, c)} \langle P, M \rangle = \sum_{i,j} P_{ij} M_{ij}$$

where $U(r, c) = \{ P \in \mathbb{R}_+^{n \times n} \mid P \mathbf{1}_n = r, \, P^T \mathbf{1}_n = c \}$.

To overcome cubic computational bottlenecks, we add an entropic regularization term:

$$d_{M,\lambda}(r, c) = \min_{P \in U(r, c)} \langle P, M \rangle - \frac{1}{\lambda} h(P)$$

where $h(P) = -\sum_{i,j} P_{ij} \ln P_{ij}$ is the Shannon entropy of the transport coupling.

---

## 2. The Sinkhorn Operator and Fixed-Point Scaling

### Theorem (Sinkhorn Matrix Scaling Representation)
*For any $\lambda > 0$, the optimal regularized transport coupling $P^\lambda$ exists, is unique, and has the explicit factorization:*

$$P^\lambda = \mathrm{diag}(u) K \mathrm{diag}(v)$$

*where $K = \exp(-\lambda M)$ is the elementwise Gibbs kernel, and $u, v \in \mathbb{R}_+^n$ are strictly positive vectors satisfying the coupled nonlinear equations:*

$$u \odot (K v) = r, \quad v \odot (K^T u) = c$$

### The Sinkhorn-Knopp Iteration
Starting from arbitrary positive vectors (e.g. $v_0 = \mathbf{1}_n$), the alternating scaling:

$$u_{k+1} = \frac{r}{K v_k}, \quad v_{k+1} = \frac{c}{K^T u_{k+1}}$$

converges linearly at an exponential rate to the unique fixed point.

---

## 3. Application to Continuous Differentiable Sorting

When the target distribution $c$ represents ordered positions ($y_1 < y_2 < \dots < y_n$) and ground cost is quadratic $M_{ij} = (x_i - y_j)^2$:
1. As $\lambda \to \infty$ ($\varepsilon \to 0^+$), the regularized coupling $P^\lambda$ converges to the exact discrete Monge-Brenier sorting permutation matrix.
2. For finite $\lambda$, $P^\lambda$ is smooth and infinitely differentiable ($C^\infty$) with respect to input values $x$.
3. The projected vector $s = n (P^\lambda)^T x$ provides a smooth, monotonic sorting operator that admits non-vanishing gradients across all coordinates, resolving the zero-gradient barrier of combinatorial sorting in machine learning.
