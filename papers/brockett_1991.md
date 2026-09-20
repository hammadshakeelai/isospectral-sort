# Dynamical Systems That Sort Lists, Diagonalize Matrices, and Solve Linear Programming Problems

**Author:** Roger W. Brockett  
**Affiliation:** Division of Applied Sciences, Harvard University, Cambridge, Massachusetts 02138  
**Journal:** *Linear Algebra and its Applications*, Volume 146, Pages 79–91, 1991.  
*(First presented at the 27th IEEE Conference on Decision and Control, Austin, Texas, December 1988)*

---

## Abstract

We study a class of smooth, continuous-time dynamical systems defined on the space of real symmetric matrices. The principal equation of interest is the double-bracket commutator flow:

$$\frac{dH}{dt} = [H, [H, N]]$$

where $H(t)$ and $N$ are symmetric $n \times n$ matrices and $[A, B] = AB - BA$ denotes the standard matrix commutator (Lie bracket). We prove that this differential equation defines an isospectral gradient flow on the adjoint orbit of the orthogonal Lie group $\mathrm{O}(n)$. When $N$ is chosen as a fixed diagonal matrix with distinct, ordered entries ($N = \mathrm{diag}(\mu_1, \dots, \mu_n)$ with $\mu_1 < \mu_2 < \dots < \mu_n$), the steady-state solution $H(\infty)$ is diagonal, with its eigenvalues arranged in strictly ascending order. Consequently, the flow sorts arbitrary lists of numbers, computes matrix eigenvalues, and solves linear programming problems in continuous physical time.

---

## 1. Introduction

Traditional algorithms in computer science treat sorting as an inherently discrete, combinatorial task requiring sequential comparisons and exchanges (such as Quicksort or Mergesort). However, in analog computing, physics, and control theory, optimization problems often admit natural representations as continuous dynamical systems that converge to the desired solution as $t \to \infty$.

In this paper, we establish that the basic algorithmic task of **sorting a list of real numbers** can be realized as the natural evolution of a continuous, autonomous differential equation on a smooth manifold.

---

## 2. The Double-Bracket Equation and Lie-Algebraic Geometry

Let $\mathcal{S}(n)$ denote the vector space of $n \times n$ real symmetric matrices, and let $\mathfrak{so}(n)$ denote the Lie algebra of $n \times n$ skew-symmetric matrices:

$$\mathfrak{so}(n) = \{ \Omega \in \mathbb{R}^{n \times n} \mid \Omega^T = -\Omega \}$$

Equip $\mathcal{S}(n)$ with the standard Frobenius inner product:

$$\langle A, B \rangle = \mathrm{Tr}(AB)$$

Given an initial symmetric matrix $H_0 \in \mathcal{S}(n)$, consider the adjoint orbit $\mathcal{O}(H_0)$ under the action of the orthogonal group $\mathrm{O}(n)$:

$$\mathcal{O}(H_0) = \{ \Theta H_0 \Theta^T \mid \Theta \in \mathrm{O}(n) \}$$

Every matrix in $\mathcal{O}(H_0)$ has the exact same set of eigenvalues as $H_0$. Thus, motion along $\mathcal{O}(H_0)$ is **strictly isospectral**.

The tangent space to the orbit $\mathcal{O}(H_0)$ at a point $H$ consists of all matrices of the form:

$$T_H \mathcal{O}(H_0) = \{ [\Omega, H] \mid \Omega \in \mathfrak{so}(n) \}$$

---

## 3. Main Theorems and Proofs

### Theorem 1 (Isospectral Flow Property)
*Let $H(t)$ satisfy $\dot{H} = [H, [H, N]]$ with $H(0) = H_0 \in \mathcal{S}(n)$ and $N \in \mathcal{S}(n)$. Then for all $t \in \mathbb{R}$:*
1. $H(t)$ is symmetric.
2. The eigenvalues of $H(t)$ are strictly independent of time $t$.
3. Any stationary point $H^*$ ($\dot{H}^* = 0$) satisfies $[H^*, N] = 0$.

**Proof:**  
Let $\Omega(t) = [N, H(t)]$. Since $H$ and $N$ are symmetric:
$$\Omega^T = (NH - HN)^T = HN - NH = -[N, H] = -\Omega$$
Hence $\Omega(t) \in \mathfrak{so}(n)$ for all $t$. The differential equation can be written as:
$$\frac{dH}{dt} = [H, -\Omega] = [\Omega, H] = \Omega H - H \Omega$$
Now consider the matrix differential equation on the orthogonal group $\mathrm{O}(n)$:
$$\dot{\Theta}(t) = \Omega(t) \Theta(t), \quad \Theta(0) = I$$
Since $\Omega(t)$ is skew-symmetric, $\Theta(t)$ remains orthogonal for all $t$ ($\Theta(t) \Theta(t)^T = I$). Differentiating $\tilde{H}(t) = \Theta(t) H_0 \Theta(t)^T$ gives:
$$\frac{d\tilde{H}}{dt} = \dot{\Theta} H_0 \Theta^T + \Theta H_0 \dot{\Theta}^T = \Omega \Theta H_0 \Theta^T + \Theta H_0 \Theta^T \Omega^T = \Omega \tilde{H} - \tilde{H} \Omega = [\Omega, \tilde{H}]$$
By uniqueness of ODE solutions, $H(t) = \Theta(t) H_0 \Theta(t)^T$. Because $H(t)$ is related to $H_0$ by an orthogonal similarity transformation, its eigenvalues are identically preserved for all $t$.

Furthermore, $\dot{H} = 0 \iff [H, [H, N]] = 0$. Taking the trace of $[H, N]^T [H, N]$:
$$\|[H, N]\|_F^2 = -\mathrm{Tr}([H, N]^2) = \mathrm{Tr}([H, [H, N]] N) = \mathrm{Tr}(\dot{H} N) = 0$$
Hence $\dot{H} = 0$ if and only if $[H, N] = 0$. $\blacksquare$

---

### Theorem 2 (Gradient Flow on the Adjoint Orbit)
*The double-bracket flow $\dot{H} = [H, [H, N]]$ is the steepest ascent gradient flow of the linear functional:*
$$\Phi(H) = \mathrm{Tr}(HN)$$
*on the Riemannian manifold $\mathcal{O}(H_0)$ equipped with the normal metric induced by the Lie algebra.*

**Proof:**  
Let $\delta H \in T_H \mathcal{O}(H_0)$ be an arbitrary tangent vector, so $\delta H = [\Omega, H]$ for some $\Omega \in \mathfrak{so}(n)$. The directional derivative of $\Phi$ along $\delta H$ is:
$$d\Phi(H)(\delta H) = \mathrm{Tr}((\delta H) N) = \mathrm{Tr}([\Omega, H] N) = \mathrm{Tr}(\Omega(HN - NH)) = \mathrm{Tr}(\Omega [H, N])$$
Under the standard inner product on $\mathfrak{so}(n)$, $\langle A, B \rangle = -\frac{1}{2}\mathrm{Tr}(AB)$, the gradient of $\Phi$ on the Lie algebra corresponds to $\Omega^* = [N, H]$. Projecting back to the manifold gives the tangent vector:
$$\mathrm{grad} \Phi(H) = [\Omega^*, H] = [[N, H], H] = [H, [H, N]]$$
Thus $\dot{H} = \mathrm{grad} \Phi(H)$ is strictly the steepest ascent gradient flow. $\blacksquare$

---

### Theorem 3 (Asymptotic Sorting via the Rearrangement Inequality)
*Suppose $N = \mathrm{diag}(\mu_1, \mu_2, \dots, \mu_n)$ has distinct ordered eigenvalues $\mu_1 < \mu_2 < \dots < \mu_n$. Let $H_0$ have distinct eigenvalues $\lambda_1 < \lambda_2 < \dots < \lambda_n$. Then:*
1. The critical points of $\Phi(H)$ on $\mathcal{O}(H_0)$ consist of all diagonal matrices whose entries are permutations of $\{\lambda_1, \dots, \lambda_n\}$. There are exactly $n!$ isolated critical points.
2. The global maximum of $\Phi(H)$ is uniquely attained at:
   $$H^* = \mathrm{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$$
3. The only asymptotically stable equilibrium point of the flow is $H^*$. For almost all initial conditions $H_0$, $H(t)$ converges as $t \to +\infty$ to $H^*$, sorting the eigenvalues in strictly increasing order.

**Proof:**  
From Theorem 1, critical points satisfy $[H^*, N] = 0$. Since $N$ is diagonal with distinct diagonal entries $\mu_i \ne \mu_j$, any matrix commuting with $N$ must itself be diagonal. Thus $H^* = \mathrm{diag}(\lambda_{\pi(1)}, \dots, \lambda_{\pi(n)})$ for some permutation $\pi \in S_n$.

Evaluating the potential at a critical point:
$$\Phi(H^*) = \sum_{i=1}^n \lambda_{\pi(i)} \mu_i$$
By the classical **Rearrangement Inequality** (Hardy, Littlewood, and Pólya, 1934):
$$\sum_{i=1}^n \lambda_{n-i+1} \mu_i \le \sum_{i=1}^n \lambda_{\pi(i)} \mu_i \le \sum_{i=1}^n \lambda_i \mu_i$$
with equality if and only if $\pi$ is the identity permutation.

To determine stability, compute the Hessian of $\Phi$ at a critical point $H^* = \mathrm{diag}(\lambda_{\pi(1)}, \dots, \lambda_{\pi(n)})$. For small variations generated by $\Omega_{ij} = E_{ij} - E_{ji} \in \mathfrak{so}(n)$, the second variation is:
$$\delta^2 \Phi = -(\lambda_{\pi(i)} - \lambda_{\pi(j)})(\mu_i - \mu_j)$$
For $H^*$ to be a local maximum, the Hessian must be negative definite, requiring:
$$(\lambda_{\pi(i)} - \lambda_{\pi(j)})(\mu_i - \mu_j) > 0 \quad \forall i < j$$
Since $\mu_i < \mu_j$ for all $i < j$, this strictly forces $\lambda_{\pi(i)} < \lambda_{\pi(j)}$ for all $i < j$, which is uniquely satisfied when $\pi$ is the identity. All other $(n! - 1)$ critical points have at least one positive eigenvalue in their Hessian and are unstable saddle points. Hence, the flow converges asymptotically to the sorted diagonal matrix $H^*$. $\blacksquare$

---

## 4. Algorithmic Applications to List Sorting

To sort an arbitrary unsorted list of real numbers $x = (x_1, x_2, \dots, x_n)$:
1. Embed $x$ into the diagonal of an initial symmetric matrix $H(0)$ with non-zero off-diagonal couplings (or rotate via an orthogonal matrix $H_0 = Q \mathrm{diag}(x) Q^T$).
2. Integrate the autonomous ODE $\dot{H} = [H, [H, N]]$ forward in continuous time $t$.
3. As $t \to +\infty$, the off-diagonal elements decay exponentially to zero, and the diagonal elements $\mathrm{diag}(H(t))$ converge to the sorted list in ascending order.
