# A Soliton Cellular Automaton

**Authors:** Daisuke Takahashi and Junkichi Satsuma  
**Affiliation:** Department of Applied Physics, Faculty of Engineering, University of Tokyo, Bunkyo-ku, Tokyo 113, Japan  
**Journal:** *Journal of the Physical Society of Japan*, Volume 59, Issue 10, Pages 3514–3519, October 1990.

---

## Abstract

We propose a 1+1 dimensional deterministic cellular automaton that exhibits exact soliton behavior. The system consists of an array of boxes where each box can contain at most one ball. The time evolution is governed by simple local rules or an equivalent "carrier" mechanism. We demonstrate that contiguous clusters of balls behave like solitons in the continuous Korteweg-de Vries (KdV) equation: their velocity is proportional to their length, and collisions between solitons of different sizes are completely elastic, preserving their individual shapes and resulting in exact phase shifts. Because larger solitons travel faster than smaller ones, an initial arbitrary configuration naturally sorts its constituent solitons by length over time.

---

## 1. Ultradiscretization and Model Definition

The Box-Ball System (BBS) can be derived through the **ultradiscretization** of the continuous Korteweg-de Vries (KdV) equation and Toda lattice. Ultradiscretization is a limiting procedure that replaces algebraic operations with the **tropical (max-plus) semiring**:

$$x \oplus y = \max(x, y), \quad x \otimes y = x + y$$

via the fundamental identity:
$$\lim_{\varepsilon \to 0^+} \varepsilon \ln(e^{A/\varepsilon} + e^{B/\varepsilon}) = \max(A, B)$$

### The Carrier Formulation
Consider an infinite 1D array of boxes indexed by $i \in \mathbb{Z}$, where state $u_i^t \in \{0, 1\}$ denotes whether box $i$ at time step $t$ is empty ($0$) or contains a ball ($1$).

At each time step $t \to t + 1$, an imaginary carrier moves from left ($i = -\infty$) to right ($i = +\infty$):
1. The carrier begins with $C = 0$ balls.
2. At box $i$:
   - If $u_i^t = 1$, the carrier picks up the ball: $u_i^{t+1} = 0$, $C \leftarrow C + 1$.
   - If $u_i^t = 0$ and $C > 0$, the carrier deposits one ball: $u_i^{t+1} = 1$, $C \leftarrow C - 1$.
   - If $u_i^t = 0$ and $C = 0$, the box remains empty: $u_i^{t+1} = 0$, $C = 0$.

---

## 2. Soliton Properties and Spatial Sorting

### Velocity Law
A contiguous cluster of $L$ balls (a soliton of size $L$) separated from other balls by at least $L$ empty boxes moves exactly $L$ boxes to the right in each time step:
$$v(L) = L$$

### Elastic Collisions and Phase Shifts
When a faster soliton of length $L_1$ overtakes a slower soliton of length $L_2$ ($L_1 > L_2$):
- During the interaction, the solitons temporarily merge.
- After emerging from the collision, both solitons fully recover their original lengths $L_1$ and $L_2$.
- The faster soliton is shifted forward by $2 L_2$ boxes, and the slower soliton is shifted backward by $2 L_2$ boxes.

### Long-Time Sorting Dynamics
Because velocity is strictly monotonic in soliton length ($v(L_1) > v(L_2) \iff L_1 > L_2$), as $t \to \infty$:
- Smaller solitons trail on the left.
- Larger solitons race ahead to the right.
- Reading the lattice from left to right yields the solitons in strictly ascending order of length:
  $$L_{(1)} \le L_{(2)} \le \dots \le L_{(n)}$$
This establishes the Box-Ball System as an ultradiscrete physical sorting automaton.
