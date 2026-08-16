"""
fea_solver_04.py

FEA From Scratch #04
Deriving the 2D Truss Element Stiffness Matrix

Purpose
-------
This is the numerical companion to the Manim video.

The solver starts from the 1D axial-bar relation and derives

    k_e = (AE/L) B^T B

where

    B = [-c, -s, c, s]

It then evaluates the matrix for an inclined member.

Units
-----
E : Pa
A : m^2
L : m
stiffness : N/m
"""

from __future__ import annotations

import numpy as np


def derive_truss_element_stiffness(
    x_i: float,
    y_i: float,
    x_j: float,
    y_j: float,
    E: float,
    A: float
):
    """
    Derive the 2D truss element stiffness matrix from B^T B.

    DOF order:
        [u_i, v_i, u_j, v_j]
    """

    dx = x_j - x_i
    dy = y_j - y_i

    L = float(np.hypot(dx, dy))

    if L <= 0.0:
        raise ValueError("Element length must be greater than zero.")

    c = dx / L
    s = dy / L

    # 1D axial strain-displacement row
    B = np.array(
        [[-c, -s, c, s]],
        dtype=float,
    )

    # 2D truss stiffness derived directly from B^T B
    k_from_B = (A * E / L) * (B.T @ B)

    # Closed-form expression
    k_direct = (A * E / L) * np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ],
        dtype=float,
    )

    return {
        "L": L,
        "c": c,
        "s": s,
        "B": B,
        "k_from_B": k_from_B,
        "k_direct": k_direct,
        "difference": k_from_B - k_direct,
        "AE_over_L": A * E / L,
    }


def solve_example():
    """
    Main worked example used in Tutorial #04.

    Inclined member:
        Node i = (0, 0) m
        Node j = (2, 1.5) m

    Material:
        E = 200 GPa

    Section:
        A = 100 mm^2
    """

    x_i, y_i = 0.0, 0.0
    x_j, y_j = 2.0, 1.5

    E = 200e9
    A = 100e-6

    result = derive_truss_element_stiffness(
        x_i,
        y_i,
        x_j,
        y_j,
        E,
        A,
    )

    return {
        **result,
        "nodes": np.array(
            [
                [x_i, y_i],
                [x_j, y_j],
            ],
            dtype=float,
        ),
        "E": E,
        "A": A,
    }


def print_report():
    r = solve_example()

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print("=" * 68)
    print("FEA FROM SCRATCH #04")
    print("2D TRUSS ELEMENT STIFFNESS MATRIX")
    print("=" * 68)

    print("\nGeometry")
    print(f"  Node i = {r['nodes'][0]}")
    print(f"  Node j = {r['nodes'][1]}")
    print(f"  L      = {r['L']:.6f} m")
    print(f"  c      = {r['c']:.6f}")
    print(f"  s      = {r['s']:.6f}")

    print("\nMaterial / section")
    print(f"  E      = {r['E'] / 1e9:.3f} GPa")
    print(f"  A      = {r['A'] * 1e6:.3f} mm^2")
    print(f"  AE/L   = {r['AE_over_L']:.3f} N/m")

    print("\nB matrix")
    print(r["B"])

    print("\nElement stiffness from (AE/L) B^T B")
    print(r["k_from_B"])

    print("\nClosed-form element stiffness")
    print(r["k_direct"])

    error = np.max(np.abs(r["difference"]))

    print("\nMaximum difference")
    print(f"  {error:.6e}")

    if np.allclose(r["k_from_B"], r["k_direct"]):
        print("\nVERIFICATION: PASSED")
    else:
        print("\nVERIFICATION: FAILED")

    print("\nFor this geometry:")
    print("  c^2 = %.6f" % (r["c"] ** 2))
    print("  cs  = %.6f" % (r["c"] * r["s"]))
    print("  s^2 = %.6f" % (r["s"] ** 2))


if __name__ == "__main__":
    print_report()
