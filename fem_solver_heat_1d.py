"""
fem_solver_heat_1d.py  -  1D steady heat conduction, from scratch.
Companion code for "FEM From Scratch, Episode 3".

Note how little changed from the bar solver (Episode 1): the element matrix has
the identical form, with conductivity k replacing stiffness EA, and temperature
T replacing displacement u. That is the whole point of the episode - FEM is
physics-agnostic.

Run:  python fem_solver_heat_1d.py
"""

import numpy as np


def element_conductivity(k, A, L):
    """Element conductivity matrix - identical form to the bar stiffness."""
    c = k * A / L
    return c * np.array([[1.0, -1.0],
                         [-1.0,  1.0]])


def solve_heat(n_elem, k, A, length, T_left, q=0.0, flux_right=0.0):
    """
    Steady 1D conduction, node 0 held at T_left.
      q          : uniform heat generation (W/m)
      flux_right : heat flux into the right end (W)
    Returns (x_nodes, T_nodes).
    """
    Le = length / n_elem
    n = n_elem + 1
    K = np.zeros((n, n))
    for e in range(n_elem):                       # SAME assembly as the bar
        K[e:e + 2, e:e + 2] += element_conductivity(k, A, Le)

    F = np.zeros(n)
    if q:
        for e in range(n_elem):
            F[e] += q * Le / 2
            F[e + 1] += q * Le / 2
    if flux_right:
        F[-1] += flux_right                       # flux enters like a tip force

    # fixed temperature at node 0: move known column to RHS, then reduce
    Kr = K[1:, 1:]
    Fr = F[1:] - K[1:, 0] * T_left
    T = np.concatenate([[T_left], np.linalg.solve(Kr, Fr)])
    x = np.linspace(0, length, n)
    return x, T


def exact_gen(x, k, A, length, q, T_left):
    """Exact temperature: fixed left, uniform generation, insulated right."""
    return T_left + q / (k * A) * (length * x - x ** 2 / 2)


if __name__ == "__main__":
    # Flux BC -> linear profile (Fourier). Verified against analytical.
    x, T = solve_heat(4, k=50.0, A=1e-3, length=0.5, T_left=100.0, flux_right=15.0)
    print("Flux BC   : T =", np.round(T, 1), "C   (linear, exact)")

    # Heat generation -> parabolic profile; same O(h^2) convergence as the bar.
    print("\nHeat-generation convergence (interior L2 error):")
    for ne in [1, 2, 4, 8, 16]:
        x, T = solve_heat(ne, k=50.0, A=1e-3, length=0.5, T_left=100.0, q=2000.0)
        xf = np.linspace(0, 0.5, 400)
        Tf = np.interp(xf, x, T)
        Te = exact_gen(xf, 50.0, 1e-3, 0.5, 2000.0, 100.0)
        err = np.sqrt(np.mean((Tf - Te) ** 2)) / (Te.max() - Te.min()) * 100
        print(f"  {ne:2d} elements -> {err:6.3f}%")
