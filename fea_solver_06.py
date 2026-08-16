"""
Engineering Insight Lab
FEA From Scratch #06 — Boundary Conditions & Solving the FEA System

Complete numerical solver for the three-node, three-member 2D truss.

This tutorial starts from the assembled 6x6 global stiffness matrix and
shows how support conditions reduce the system before solving.

Units:
    Geometry       : m
    Area           : m^2
    Young's modulus: Pa
    Force          : N
    Stiffness      : N/m
    Displacement   : m

Global DOF order:
    [u1, v1, u2, v2, u3, v3]

Supports:
    Node 1: pin    -> u1 = 0, v1 = 0
    Node 2: roller -> v2 = 0

Unknown/free DOFs:
    [u2, u3, v3] -> global indices [2, 4, 5]
"""

from __future__ import annotations

import numpy as np


# ============================================================
# MODEL
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

A = 100.0e-6
E = 200.0e9
P = -10_000.0


# ============================================================
# ELEMENT STIFFNESS
# ============================================================

def element_stiffness(node_i, node_j):
    dx = float(node_j[0] - node_i[0])
    dy = float(node_j[1] - node_i[1])
    L = float(np.hypot(dx, dy))

    if L <= 0:
        raise ValueError("Zero-length truss element.")

    c = dx / L
    s = dy / L
    factor = A * E / L

    ke = factor * np.array(
        [
            [c*c,  c*s, -c*c, -c*s],
            [c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s],
        ],
        dtype=float,
    )

    return ke, L, c, s, factor


# ============================================================
# GLOBAL ASSEMBLY
# ============================================================

def assemble_global_stiffness():
    K = np.zeros((6, 6), dtype=float)
    elements = []

    for e, (i, j) in enumerate(ELEMENTS, start=1):
        ke, L, c, s, factor = element_stiffness(
            NODES[i], NODES[j]
        )

        dofs = np.array(
            [2*i, 2*i+1, 2*j, 2*j+1],
            dtype=int,
        )

        contribution = np.zeros((6, 6), dtype=float)

        for a, I in enumerate(dofs):
            for b, J in enumerate(dofs):
                contribution[I, J] = ke[a, b]
                K[I, J] += ke[a, b]

        elements.append(
            {
                "element": e,
                "nodes": (i+1, j+1),
                "dofs": dofs,
                "ke": ke,
                "global_contribution": contribution,
                "length": L,
                "c": c,
                "s": s,
                "factor": factor,
            }
        )

    return K, elements


# ============================================================
# BOUNDARY CONDITIONS + SOLUTION
# ============================================================

def solve_fea():
    K, elements = assemble_global_stiffness()

    # Global force vector:
    # [Fx1, Fy1, Fx2, Fy2, Fx3, Fy3]
    F = np.zeros(6, dtype=float)
    F[5] = P

    # Prescribed DOFs from supports:
    # u1 = 0, v1 = 0, v2 = 0
    constrained = np.array([0, 1, 3], dtype=int)

    # Unknown DOFs:
    # u2, u3, v3
    free = np.array([2, 4, 5], dtype=int)

    prescribed_values = np.zeros(len(constrained))

    # Partition the global system:
    #
    # [Kff Kfc] [Uf] = [Ff]
    # [Kcf Kcc] [Uc]   [Fc]
    #
    # Since Uc = 0:
    #     Kff Uf = Ff
    Kff = K[np.ix_(free, free)]
    Kfc = K[np.ix_(free, constrained)]
    Kcf = K[np.ix_(constrained, free)]
    Kcc = K[np.ix_(constrained, constrained)]

    Ff = F[free]
    Fc = F[constrained]

    # General form:
    # Kff Uf = Ff - Kfc Uc
    rhs = Ff - Kfc @ prescribed_values

    Uf = np.linalg.solve(Kff, rhs)

    # Reconstruct the complete displacement vector.
    U = np.zeros(6, dtype=float)
    U[constrained] = prescribed_values
    U[free] = Uf

    # Reactions:
    # R = KU - F
    reactions = K @ U - F

    # Element response.
    for item in elements:
        dofs = item["dofs"]
        c = item["c"]
        s = item["s"]
        L = item["length"]

        ue = U[dofs]

        axial_extension = np.dot(
            np.array([-c, -s, c, s]),
            ue,
        )

        strain = axial_extension / L
        stress = E * strain
        axial_force = A * stress

        item["ue"] = ue
        item["extension"] = axial_extension
        item["strain"] = strain
        item["stress"] = stress
        item["axial_force"] = axial_force

    # Checks.
    force_balance_vector = K @ U - F
    total_vertical_reaction = reactions[1] + reactions[3]

    return {
        "nodes": NODES.copy(),
        "elements": elements,
        "A": A,
        "E": E,
        "load": P,
        "K": K,
        "F": F,
        "constrained_dofs": constrained,
        "free_dofs": free,
        "prescribed_values": prescribed_values,
        "Kff": Kff,
        "Kfc": Kfc,
        "Kcf": Kcf,
        "Kcc": Kcc,
        "Ff": Ff,
        "Fc": Fc,
        "rhs": rhs,
        "Uf": Uf,
        "U": U,
        "reactions": reactions,
        "force_balance_vector": force_balance_vector,
        "total_vertical_reaction": total_vertical_reaction,
        "element_results": elements,
    }


def solve_truss():
    """Compatibility alias for the series."""
    return solve_fea()


# ============================================================
# COMMAND-LINE CHECK
# ============================================================

if __name__ == "__main__":
    result = solve_fea()

    np.set_printoptions(precision=9, suppress=True)

    print("\n=== GLOBAL STIFFNESS MATRIX K [10^6 N/m] ===")
    print(result["K"] / 1.0e6)

    print("\n=== CONSTRAINED DOFs ===")
    print(result["constrained_dofs"], " -> u1, v1, v2")

    print("\n=== FREE DOFs ===")
    print(result["free_dofs"], " -> u2, u3, v3")

    print("\n=== REDUCED MATRIX Kff [10^6 N/m] ===")
    print(result["Kff"] / 1.0e6)

    print("\n=== REDUCED FORCE VECTOR Ff [N] ===")
    print(result["Ff"])

    print("\n=== UNKNOWN DISPLACEMENTS [mm] ===")
    print(result["Uf"] * 1000.0)

    print("\n=== COMPLETE DISPLACEMENT VECTOR [mm] ===")
    print(result["U"] * 1000.0)

    print("\n=== REACTIONS [kN] ===")
    print(result["reactions"] / 1000.0)

    print("\n=== SUPPORT REACTION VECTOR [kN] ===")
    print(result["reactions"] / 1000.0)

    print("\n=== MEMBER FORCES [kN] ===")
    for item in result["element_results"]:
        print(
            f"Element {item['element']}: "
            f"N = {item['axial_force']/1000:.6f} kN, "
            f"stress = {item['stress']/1e6:.6f} MPa"
        )
