"""
Engineering Insight Lab
FEA From Scratch #05 — Global Stiffness Matrix Assembly

Numerical solver for the same three-node, three-member 2D truss used in
Tutorial #03 and Tutorial #04.

Units:
    Geometry : m
    Area     : m^2
    E        : Pa
    Force    : N
    Stiffness: N/m
    Displacement: m

Global DOF order:
    [u1, v1, u2, v2, u3, v3]

Element connectivity:
    e1: Node 1 -> Node 2  => [0, 1, 2, 3]
    e2: Node 1 -> Node 3  => [0, 1, 4, 5]
    e3: Node 2 -> Node 3  => [2, 3, 4, 5]
"""

from __future__ import annotations

import numpy as np


# ============================================================
# MODEL DATA
# ============================================================

NODES = np.array(
    [
        [0.0, 0.0],   # Node 1
        [2.0, 0.0],   # Node 2
        [1.0, 1.5],   # Node 3
    ],
    dtype=float,
)

ELEMENTS = (
    (0, 1),  # Element 1
    (0, 2),  # Element 2
    (1, 2),  # Element 3
)

A = 100.0e-6       # 100 mm^2 = 100e-6 m^2
E = 200.0e9        # 200 GPa
LOAD = -10_000.0   # 10 kN downward at Node 3


# ============================================================
# ELEMENT STIFFNESS
# ============================================================

def element_stiffness(
    node_i: np.ndarray,
    node_j: np.ndarray,
    area: float = A,
    youngs_modulus: float = E,
):
    """Return the 4x4 global-coordinate stiffness matrix and geometry."""

    dx = float(node_j[0] - node_i[0])
    dy = float(node_j[1] - node_i[1])

    length = float(np.hypot(dx, dy))
    if length <= 0.0:
        raise ValueError("Element length must be greater than zero.")

    c = dx / length
    s = dy / length
    factor = area * youngs_modulus / length

    ke = factor * np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ],
        dtype=float,
    )

    return ke, length, c, s, factor


# ============================================================
# ASSEMBLY
# ============================================================

def assemble_global_stiffness():
    """Assemble the complete 6x6 structural stiffness matrix."""

    K = np.zeros((6, 6), dtype=float)
    element_data = []

    for element_number, (i, j) in enumerate(ELEMENTS, start=1):
        ke, length, c, s, factor = element_stiffness(
            NODES[i], NODES[j]
        )

        dof_map = [
            2 * i,
            2 * i + 1,
            2 * j,
            2 * j + 1,
        ]

        K_before = K.copy()

        for a, I in enumerate(dof_map):
            for b, J in enumerate(dof_map):
                K[I, J] += ke[a, b]

        K_after = K.copy()

        element_data.append(
            {
                "element": element_number,
                "nodes": (i + 1, j + 1),
                "dof_map": dof_map,
                "ke": ke,
                "length": length,
                "c": c,
                "s": s,
                "factor": factor,
                "K_before": K_before,
                "K_after": K_after,
            }
        )

    return K, element_data


# ============================================================
# SOLVE
# ============================================================

def solve_truss():
    """Assemble and solve the three-member triangular truss."""

    K, element_data = assemble_global_stiffness()

    # Global force vector:
    # [Fx1, Fy1, Fx2, Fy2, Fx3, Fy3]
    F = np.zeros(6, dtype=float)
    F[5] = LOAD

    # Supports:
    # Node 1 pin  -> u1 = 0, v1 = 0
    # Node 2 roller -> v2 = 0
    constrained_dofs = np.array([0, 1, 3], dtype=int)
    free_dofs = np.array([2, 4, 5], dtype=int)

    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]

    U = np.zeros(6, dtype=float)
    U[free_dofs] = np.linalg.solve(K_reduced, F_reduced)

    # Reactions from R = KU - F
    reactions = K @ U - F

    # Element axial force, strain and stress
    for item in element_data:
        mp = item["dof_map"]
        ke = item["ke"]
        c = item["c"]
        s = item["s"]
        length = item["length"]

        ue = U[mp]

        # Positive = tension; negative = compression.
        axial_force = (
            A * E / length
            * np.dot(
                np.array([-c, -s, c, s]),
                ue,
            )
        )

        axial_strain = (
            np.dot(
                np.array([-c, -s, c, s]),
                ue,
            )
            / length
        )

        axial_stress = E * axial_strain

        item["element_displacement"] = ue
        item["axial_force"] = axial_force
        item["strain"] = axial_strain
        item["stress"] = axial_stress

    # Useful assembly contributions for animation.
    # Each contribution is a 6x6 matrix with the element's 4x4
    # matrix placed in the appropriate global DOF locations.
    for item in element_data:
        contribution = np.zeros((6, 6), dtype=float)
        mp = item["dof_map"]
        ke = item["ke"]

        for a, I in enumerate(mp):
            for b, J in enumerate(mp):
                contribution[I, J] = ke[a, b]

        item["global_contribution"] = contribution

    return {
        "nodes": NODES.copy(),
        "elements": ELEMENTS,
        "A": A,
        "E": E,
        "load": LOAD,
        "K": K,
        "F": F,
        "U": U,
        "K_reduced": K_reduced,
        "F_reduced": F_reduced,
        "constrained_dofs": constrained_dofs,
        "free_dofs": free_dofs,
        "reactions": reactions,
        "element_results": element_data,
    }


if __name__ == "__main__":
    result = solve_truss()

    np.set_printoptions(precision=6, suppress=True)

    print("\nGLOBAL STIFFNESS MATRIX K [N/m]")
    print(result["K"])

    print("\nGLOBAL STIFFNESS MATRIX K [10^6 N/m]")
    print(result["K"] / 1.0e6)

    print("\nGLOBAL FORCE VECTOR F [N]")
    print(result["F"])

    print("\nDISPLACEMENT VECTOR U [mm]")
    print(result["U"] * 1000.0)

    print("\nREACTIONS [kN]")
    print(result["reactions"] / 1000.0)

    print("\nELEMENT RESULTS")
    for item in result["element_results"]:
        print(
            f"Element {item['element']}: "
            f"DOFs={item['dof_map']}, "
            f"L={item['length']:.4f} m, "
            f"c={item['c']:.4f}, "
            f"s={item['s']:.4f}, "
            f"N={item['axial_force']/1000:.6f} kN"
        )
