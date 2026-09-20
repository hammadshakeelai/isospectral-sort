# Fast Differentiable Sorting and Ranking

**Authors:** Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga  
**Affiliation:** Google Research, Brain Team, Paris / Zurich  
**Conference:** *International Conference on Machine Learning (ICML 2020)*, PMLR 119:950–959.

---

## Abstract

Sorting and ranking operations are foundational building blocks throughout algorithmic computing. In machine learning, however, discrete sorting behaves as a piecewise constant step function whose gradients vanish almost everywhere and are undefined at ties and transitions, preventing end-to-end backpropagation in deep neural networks.

We propose a continuous relaxation framework that casts sorting and ranking as regularized optimal transport problems and projections onto the permutahedron. By using strongly convex regularizers (including entropy and quadratic regularization), we derive continuous sorting operators that are strictly monotonic, differentiable, and computable in $O(n \log n)$ or fast GPU matrix iterations. We show that optimal transport continuously interpolates between identity sorting and soft differentiable permutations.

---

## 1. Differentiable Sorting as Projection onto the Permutahedron

Let $w = (1, 2, \dots, n)$ be the vector of canonical ascending ranks. The permutahedron $\mathcal{P}_w$ is the convex hull of all permutations of $w$:

$$\mathcal{P}_w = \mathrm{conv}(\{ P w \mid P \in \mathcal{P}_n \})$$

The discrete sorting problem on an input vector $\theta \in \mathbb{R}^n$ can be formulated as a linear program over the permutahedron:

$$s(\theta) = \operatorname{argmax}_{y \in \mathcal{P}_w} \langle y, \theta \rangle$$

By the Rearrangement Inequality, $s(\theta)$ is uniquely maximized when $y$ has the same ordering as $\theta$.

---

## 2. Strongly Convex Regularization and Smooth Ranking

To obtain differentiable sorting operators, we introduce a strongly convex regularizer $\Omega$:

$$s_\Omega(\theta) = \operatorname{argmax}_{y \in \mathcal{P}_w} \langle y, \theta \rangle - \Omega(y)$$

### Theorem (Differentiability of Regularized Sorting)
*If $\Omega$ is $\gamma$-strongly convex, then:*
1. The regularized sorting operator $s_\Omega(\theta)$ is unique, strictly monotonic, and Lipschitz continuous.
2. The Jacobian $\nabla s_\Omega(\theta)$ exists almost everywhere, and is positive semi-definite.
3. For entropic regularization $\Omega(y) = \varepsilon \sum_i y_i \ln y_i$, the operator coincides with the marginal expectation of entropic optimal transport (Sinkhorn algorithm).

---

## 3. Connection to Monge-Brenier Transport

When relaxed over the Birkhoff polytope of doubly stochastic matrices $\mathcal{B}_n$:

$$\min_{P \in \mathcal{B}_n} \langle P, C(\theta, w) \rangle - \varepsilon H(P)$$

the continuous transport plan $P_\varepsilon(\theta)$ gives an explicit differentiable permutation matrix. Multiplying by the input vector:

$$\hat{\theta}_{\mathrm{sorted}} = P_\varepsilon(\theta)^T \theta$$

yields an infinitely differentiable continuous sorting algorithm with exact asymptotic convergence to standard sort as $\varepsilon \to 0^+$.
