"""
fem_solver_1d.py  -  1D Finite Element solver, from scratch, no black boxes.
Companion code for "FEM From Scratch, Episode 1".

Solves a fixed-free 1D bar (E, A, length) under a tip point load and/or a
uniform distributed load, using linear 2-node bar elements.

Run:  python fem_solver_1d.py
"""

import numpy as np


def element_stiffness(E, A, L):
    """Stiffness matrix of one 2-node bar element: (EA/L)[[1,-1],[-1,1]]."""
    k = E * A / L
    return k * np.array([[1.0, -1.0],
                         [-1.0,  1.0]])


def solve_bar(n_elem, E, A, length, tip_load=0.0, dist_load=0.0):
    """
    Fixed-free bar, node 0 fixed (u=0).
      tip_load  : point load P at the free end (N)
      dist_load : uniform distributed load f (N/m)
    Returns (x_nodes, u_nodes).
    """
    Le = length / n_elem
    n = n_elem + 1
    K = np.zeros((n, n))

    # ---- ASSEMBLY: overlap-and-add at shared nodes ----
    for e in range(n_elem):
        K[e:e + 2, e:e + 2] += element_stiffness(E, A, Le)

    # ---- LOADS ----
    F = np.zeros(n)
    if tip_load:
        F[-1] += tip_load
    if dist_load:                       # consistent lumping: f*Le/2 to each node
        for e in range(n_elem):
            F[e] += dist_load * Le / 2
            F[e + 1] += dist_load * Le / 2

    # ---- BOUNDARY CONDITION: node 0 fixed -> drop row/col 0 ----
    Kr, Fr = K[1:, 1:], F[1:]
    u = np.concatenate([[0.0], np.linalg.solve(Kr, Fr)])
    x = np.linspace(0, length, n)
    return x, u


def exact_dist(x, E, A, length, f):
    """Exact displacement under uniform load (parabola), for comparison."""
    return f / (2 * E * A) * (2 * length * x - x ** 2)


if __name__ == "__main__":
    # Point-load demo: linear FEM is exact here.
    x, u = solve_bar(3, E=1.0, A=1.0, length=3.0, tip_load=1.0)
    print("Point load   : u =", np.round(u, 4), " tip =", round(u[-1], 4),
          " (exact PL/EA = 3.0)")

    # Distributed-load demo: watch interior error shrink ~O(h^2).
    print("\nDistributed load convergence (interior L2 error):")
    for ne in [1, 2, 4, 8, 16]:
        x, u = solve_bar(ne, E=1.0, A=1.0, length=3.0, dist_load=1.0)
        xf = np.linspace(0, 3, 400)
        uf = np.interp(xf, x, u)
        ue = exact_dist(xf, 1.0, 1.0, 3.0, 1.0)
        err = np.sqrt(np.mean((uf - ue) ** 2)) / ue.max() * 100
        print(f"  {ne:2d} elements -> {err:6.3f}%")
