# Optimal Transport, Monotone Rearrangements, and Differentiable Ranking

**Foundational Works:**
- Yann Brenier (1991): *"Polar factorization and monotone rearrangement of vector-valued functions."* *Comm. Pure Appl. Math.*, 44(4):375–417.
- Marco Cuturi (2013): *"Sinkhorn Distances: Lightspeed Computation of Optimal Transport."* *Advances in Neural Information Processing Systems (NeurIPS 2013)*.
- Mathieu Blondel et al. (2020): *"Fast Differentiable Sorting and Ranking."* *International Conference on Machine Learning (ICML 2020)*.

---

## 1. Monge-Kantorovich Formulation of 1D Sorting

Let $\mu = \frac{1}{n}\sum_{i=1}^n \delta_{x_i}$ be an empirical probability measure on $\mathbb{R}$ representing the unsorted input values $x = (x_1, \dots, x_n)$. Let $\nu = \frac{1}{n}\sum_{j=1}^n \delta_{y_j}$ be a reference probability measure representing ordered ranks $y_1 < y_2 < \dots < y_n$ (e.g. $y_j = j$).

The Kantorovich optimal transport problem seeks a transport coupling $P \in \mathbb{R}_{+}^{n \times n}$ solving:

$$\min_{P \in \mathcal{U}(a, b)} \langle P, C \rangle = \sum_{i=1}^n \sum_{j=1}^n P_{ij} C_{ij}$$

subject to marginal constraints:
$$P \mathbf{1}_n = \frac{1}{n}\mathbf{1}_n, \quad P^T \mathbf{1}_n = \frac{1}{n}\mathbf{1}_n$$
where $\mathcal{U}(a, b)$ is the Birkhoff polytope of doubly stochastic matrices scaled by $1/n$, and $C_{ij} = c(x_i, y_j)$ is the transport ground cost.

---

## 2. Brenier's Polar Factorization Theorem

### Theorem (Brenier, 1991; 1D Monotonicity)
*Let $c(x, y) = h(x - y)$ where $h$ is strictly convex (e.g. quadratic cost $c(x, y) = (x - y)^2$). Then the optimal transport map $T^*: \mathrm{supp}(\mu) \to \mathrm{supp}(\nu)$ exists, is unique, and is strictly monotonically increasing:*
$$x_i < x_k \implies T^*(x_i) < T^*(x_k)$$

### Connection to Sorting
When $\nu$ is an ordered target ($y_1 < y_2 < \dots < y_n$), the monotonicity of $T^*$ implies that the optimal assignment matrix $P^*$ is a permutation matrix that maps the smallest element of $x$ to $y_1$, the second smallest to $y_2$, and so forth.

Equivalently, expanding the quadratic cost:
$$(x_i - y_j)^2 = x_i^2 - 2 x_i y_j + y_j^2$$
Because $\sum_j P_{ij} x_i^2 = x_i^2 / n$ and $\sum_i P_{ij} y_j^2 = y_j^2 / n$ are invariant under the marginal constraints, minimizing $\langle P, C \rangle$ is mathematically equivalent to maximizing:
$$\sum_{i=1}^n \sum_{j=1}^n P_{ij} x_i y_j$$
By the **Rearrangement Inequality**, this is maximized when $x$ and $y$ are sorted in the same direction.

---

## 3. Entropic Regularization and the Sinkhorn Operator

Traditional discrete sorting is a piecewise constant step function with zero derivatives almost everywhere and discontinuities at every swap, preventing gradient-based backpropagation in deep neural networks.

To make sorting differentiable, Cuturi (2013) introduces entropic regularization:
$$\min_{P \in \mathcal{U}} \langle P, C \rangle - \varepsilon H(P)$$
where $H(P) = -\sum_{i,j} P_{ij}(\ln P_{ij} - 1)$ is the Shannon entropy.

The optimal solution is given by:
$$P_\varepsilon = \mathrm{diag}(u) K \mathrm{diag}(v), \quad K_{ij} = \exp(-C_{ij}/\varepsilon)$$
where $u, v \in \mathbb{R}_+^n$ are computed via Sinkhorn-Knopp fixed-point iterations:
$$u \leftarrow \frac{a}{K v}, \quad v \leftarrow \frac{b}{K^T u}$$

### Properties:
1. **Infinite Differentiability:** $P_\varepsilon(x)$ is $C^\infty$ with respect to the input values $x$.
2. **Asymptotic Exactness:** As $\varepsilon \to 0^+$, $P_\varepsilon$ converges to the exact discrete sorting permutation matrix $P^*$.
3. **Soft Sorting:** The vector $s = n P_\varepsilon^T x$ is a differentiable relaxation of the sorted array $\mathrm{sort}(x)$.
