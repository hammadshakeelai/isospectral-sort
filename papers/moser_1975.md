# Finitely Many Points on the Line Under the Influence of an Exponential Potential: An Integrable System

**Author:** Jürgen Moser  
**Affiliation:** Courant Institute of Mathematical Sciences, New York University, New York, NY 10012  
**Publication:** *Dynamical Systems, Theory and Applications*, Lecture Notes in Physics, Vol. 38, pp. 467–497, Springer-Verlag, Berlin/Heidelberg, 1975.

---

## Abstract

We study the classical dynamics of $n$ particles on a one-dimensional real line interacting through exponential repulsive nearest-neighbor potentials (the non-periodic Toda lattice). Using the transformation introduced by Hermann Flaschka, the equations of motion are cast into a Lax pair differential equation $\dot{L} = [B, L]$ on symmetric tridiagonal Jacobi matrices. We prove that the system is completely integrable in the sense of Liouville. 

In the asymptotic limit $t \to +\infty$, the particle interactions decay exponentially ($a_k \to 0$), and the diagonal elements $b_k(t)$ decouple and converge to the system's conserved eigenvalues arranged in strictly descending order. Conversely, in the backward time limit $t \to -\infty$, the diagonal elements converge to the eigenvalues arranged in strictly ascending order. Thus, the physical scattering of particles in an exponential potential naturally executes a continuous sorting algorithm on the eigenvalues of the system.

---

## 1. Physical Model and Equations of Motion

Consider $n$ unit-mass particles on the real line with coordinates $q_1 < q_2 < \dots < q_n$ and momenta $p_1, \dots, p_n$. The Hamiltonian is given by:

$$H(p, q) = \frac{1}{2}\sum_{k=1}^n p_k^2 + \sum_{k=1}^{n-1} e^{-(q_{k+1} - q_k)}$$

The Hamilton equations of motion are:

$$\dot{q}_k = \frac{\partial H}{\partial p_k} = p_k$$
$$\dot{p}_k = -\frac{\partial H}{\partial q_k} = e^{-(q_k - q_{k-1})} - e^{-(q_{k+1} - q_k)}$$

with boundary conventions $q_0 = -\infty$ and $q_{n+1} = +\infty$.

---

## 2. Flaschka Variables and the Lax Formulation

In 1974, Hermann Flaschka introduced the change of coordinates:

$$a_k = \frac{1}{2} e^{-(q_{k+1} - q_k)/2}, \quad k = 1, 2, \dots, n-1$$
$$b_k = -\frac{1}{2} p_k, \quad k = 1, 2, \dots, n$$

In these coordinates, the phase space equations become:

$$\dot{a}_k = a_k(b_{k+1} - b_k), \quad k = 1, \dots, n-1$$
$$\dot{b}_k = 2(a_k^2 - a_{k-1}^2), \quad k = 1, \dots, n$$

with $a_0 = a_n = 0$.

Define the real symmetric tridiagonal Jacobi matrix $L$ and the skew-symmetric matrix $B$:

$$L = \begin{pmatrix}
b_1 & a_1 & 0 & \dots & 0 \\
a_1 & b_2 & a_2 & \dots & 0 \\
0 & a_2 & b_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & a_{n-1} \\
0 & 0 & 0 & a_{n-1} & b_n
\end{pmatrix}, \quad
B = \begin{pmatrix}
0 & a_1 & 0 & \dots & 0 \\
-a_1 & 0 & a_2 & \dots & 0 \\
0 & -a_2 & 0 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & a_{n-1} \\
0 & 0 & 0 & -a_{n-1} & 0
\end{pmatrix}$$

The Flaschka system is completely equivalent to the Lax equation:

$$\frac{dL}{dt} = [B, L] = BL - LB$$

---

## 3. Asymptotic Scattering and Eigenvalue Sorting

### Theorem (Moser's Asymptotic Scattering Theorem)
*Let $L(t)$ evolve according to $\dot{L} = [B, L]$. Then:*
1. The eigenvalues $\lambda_1 < \lambda_2 < \dots < \lambda_n$ of $L(t)$ are strictly invariant for all $t$.
2. As $t \to +\infty$:
   $$a_k(t) \to 0, \quad k = 1, \dots, n-1$$
   $$b_k(t) \to \lambda_{n-k+1}, \quad k = 1, \dots, n$$
   That is, the diagonal entries converge to the eigenvalues in **descending order**:
   $$\lim_{t \to +\infty} \operatorname{diag}(L(t)) = (\lambda_n, \lambda_{n-1}, \dots, \lambda_1)$$
3. As $t \to -\infty$:
   $$\lim_{t \to -\infty} \operatorname{diag}(L(t)) = (\lambda_1, \lambda_2, \dots, \lambda_n)$$
   That is, the diagonal entries converge to the eigenvalues in **ascending order**.

### Physical Intuition
Because the particles repel each other, no two particles can occupy the same position. As $t \to +\infty$, the particles separate indefinitely ($q_{k+1} - q_k \to \infty$), causing the inter-particle forces $a_k \propto e^{-(q_{k+1}-q_k)/2}$ to vanish. 

The particles decouple into free, non-interacting particles whose asymptotic velocities are ordered by physical speed: the fastest particle escapes furthest to the right, and the slowest particle lags behind on the left.
