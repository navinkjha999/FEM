"""
Engineering Insight Lab
FEA From Scratch #03 — 2D Truss Analysis

PURE NUMERICAL SOLVER
---------------------

This file contains NO Manim code.

Problem
-------
Symmetric triangular truss:

                    Node 3
                      ●
                     / \
                    /   \
                   /     \
                  /       \
                 ●─────────●
              Node 1     Node 2

Node 1:
    Pin support
    ux = 0
    uy = 0

Node 2:
    Roller support
    uy = 0

Node 3:
    Vertical load = 10 kN downward

Geometry:
    Node 1 = (0, 0) m
    Node 2 = (2, 0) m
    Node 3 = (1, 1.5) m

Material:
    E = 200 GPa

Area:
    A = 100 mm²

The solver performs:

    1. Node definition
    2. Element connectivity
    3. Element geometry
    4. Direction cosines
    5. 2D truss element stiffness matrix
    6. Global stiffness assembly
    7. Boundary conditions
    8. Solution of KU = F
    9. Reactions
    10. Element strain
    11. Element stress
    12. Element axial force
"""

from __future__ import annotations

import numpy as np


# ============================================================
# PROBLEM DEFINITION
# ============================================================

E = 200e9                 # Pa
AREA = 100e-6             # m² = 100 mm²

NODES = np.array(
    [
        [0.0, 0.0],       # Node 1
        [2.0, 0.0],       # Node 2
        [1.0, 1.5],       # Node 3
    ],
    dtype=float,
)

# Element connectivity.
#
# Element 1: Node 1 -> Node 2
# Element 2: Node 1 -> Node 3
# Element 3: Node 2 -> Node 3

ELEMENTS = [
    (0, 1),
    (0, 2),
    (1, 2),
]

# Global force vector.
#
# DOF ordering:
# Node 1 -> ux1, uy1
# Node 2 -> ux2, uy2
# Node 3 -> ux3, uy3

FORCES = np.zeros(6)

# 10 kN downward at Node 3.
FORCES[5] = -10_000.0


# ============================================================
# ELEMENT GEOMETRY
# ============================================================

def element_geometry(
    node_i: int,
    node_j: int,
    nodes: np.ndarray = NODES,
) -> tuple[float, float, float]:
    """
    Calculate:

        L = element length
        c = cos(theta)
        s = sin(theta)
    """

    xi, yi = nodes[node_i]
    xj, yj = nodes[node_j]

    dx = xj - xi
    dy = yj - yi

    length = np.sqrt(
        dx**2 + dy**2
    )

    if length <= 0:
        raise ValueError(
            "Element length must be greater than zero."
        )

    c = dx / length
    s = dy / length

    return length, c, s


# ============================================================
# 2D TRUSS ELEMENT STIFFNESS MATRIX
# ============================================================

def truss_element_stiffness(
    E: float,
    A: float,
    length: float,
    c: float,
    s: float,
) -> np.ndarray:
    """
    2D truss element stiffness matrix in global coordinates.

          AE/L *
          [ c²    cs   -c²   -cs ]
          [ cs    s²   -cs   -s² ]
          [-c²   -cs    c²    cs ]
          [-cs   -s²    cs    s² ]
    """

    factor = E * A / length

    return factor * np.array(
        [
            [
                c**2,
                c * s,
                -c**2,
                -c * s,
            ],
            [
                c * s,
                s**2,
                -c * s,
                -s**2,
            ],
            [
                -c**2,
                -c * s,
                c**2,
                c * s,
            ],
            [
                -c * s,
                -s**2,
                c * s,
                s**2,
            ],
        ]
    )


# ============================================================
# GLOBAL DOF MAPPING
# ============================================================

def element_dofs(
    node_i: int,
    node_j: int,
) -> list[int]:
    """
    Return global DOF numbers for a 2D truss element.

    Node i:
        ux_i = 2*i
        uy_i = 2*i + 1

    Node j:
        ux_j = 2*j
        uy_j = 2*j + 1
    """

    return [
        2 * node_i,
        2 * node_i + 1,
        2 * node_j,
        2 * node_j + 1,
    ]


# ============================================================
# GLOBAL STIFFNESS ASSEMBLY
# ============================================================

def assemble_global_stiffness(
    nodes: np.ndarray = NODES,
    elements: list[tuple[int, int]] = ELEMENTS,
    E: float = E,
    A: float = AREA,
) -> tuple[np.ndarray, list[dict]]:
    """
    Assemble all element stiffness matrices
    into the global stiffness matrix.
    """

    number_of_nodes = len(nodes)

    number_of_dofs = (
        2 * number_of_nodes
    )

    K = np.zeros(
        (
            number_of_dofs,
            number_of_dofs,
        )
    )

    element_data = []

    for element_number, (node_i, node_j) in enumerate(
        elements,
        start=1,
    ):

        length, c, s = element_geometry(
            node_i,
            node_j,
            nodes,
        )

        ke = truss_element_stiffness(
            E,
            A,
            length,
            c,
            s,
        )

        dofs = element_dofs(
            node_i,
            node_j,
        )

        # ----------------------------------------------------
        # Assembly
        # ----------------------------------------------------

        for a in range(4):

            for b in range(4):

                K[
                    dofs[a],
                    dofs[b]
                ] += ke[a, b]

        element_data.append(
            {
                "element": element_number,
                "node_i": node_i,
                "node_j": node_j,
                "length": length,
                "c": c,
                "s": s,
                "dofs": dofs,
                "ke": ke,
            }
        )

    return K, element_data


# ============================================================
# APPLY BOUNDARY CONDITIONS
# ============================================================

def solve_with_boundary_conditions(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve:

        K U = F

    after applying prescribed zero displacements.

    Returns:

        U          = complete displacement vector
        reactions  = reaction vector
        free_dofs  = unknown DOFs
    """

    total_dofs = len(F)

    all_dofs = np.arange(
        total_dofs
    )

    fixed_dofs = np.array(
        fixed_dofs,
        dtype=int,
    )

    free_dofs = np.array(
        [
            dof
            for dof in all_dofs
            if dof not in fixed_dofs
        ],
        dtype=int,
    )

    # --------------------------------------------------------
    # Reduced system
    # --------------------------------------------------------

    K_ff = K[
        np.ix_(
            free_dofs,
            free_dofs,
        )
    ]

    F_f = F[
        free_dofs
    ]

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    U = np.zeros(
        total_dofs
    )

    U[free_dofs] = np.linalg.solve(
        K_ff,
        F_f,
    )

    # --------------------------------------------------------
    # Reactions
    # --------------------------------------------------------

    reactions = (
        K @ U - F
    )

    return (
        U,
        reactions,
        free_dofs,
    )


# ============================================================
# ELEMENT POST-PROCESSING
# ============================================================

def calculate_element_results(
    U: np.ndarray,
    element_data: list[dict],
    E: float,
    A: float,
) -> list[dict]:
    """
    Calculate:

        axial deformation
        strain
        stress
        axial force

    for each truss member.
    """

    results = []

    for data in element_data:

        dofs = data["dofs"]
        c = data["c"]
        s = data["s"]
        L = data["length"]

        ue = U[
            dofs
        ]

        # ----------------------------------------------------
        # Axial deformation
        # ----------------------------------------------------

        delta = (
            -c * ue[0]
            -s * ue[1]
            +c * ue[2]
            +s * ue[3]
        )

        # ----------------------------------------------------
        # Strain
        # ----------------------------------------------------

        strain = (
            delta / L
        )

        # ----------------------------------------------------
        # Stress
        # ----------------------------------------------------

        stress = (
            E * strain
        )

        # ----------------------------------------------------
        # Axial force
        # ----------------------------------------------------

        axial_force = (
            A * stress
        )

        results.append(
            {
                **data,
                "displacement": ue,
                "axial_deformation": delta,
                "strain": strain,
                "stress": stress,
                "axial_force": axial_force,
            }
        )

    return results


# ============================================================
# COMPLETE TRUSS SOLUTION
# ============================================================

def solve_truss() -> dict:
    """
    Complete FEA solution.
    """

    # --------------------------------------------------------
    # Assemble K
    # --------------------------------------------------------

    K, element_data = (
        assemble_global_stiffness()
    )

    # --------------------------------------------------------
    # Boundary conditions
    #
    # Node 1:
    #     ux1 = 0
    #     uy1 = 0
    #
    # Node 2:
    #     uy2 = 0
    #
    # DOFs:
    #
    # 0 = ux1
    # 1 = uy1
    # 2 = ux2
    # 3 = uy2
    # 4 = ux3
    # 5 = uy3
    # --------------------------------------------------------

    fixed_dofs = [
        0,
        1,
        3,
    ]

    U, reactions, free_dofs = (
        solve_with_boundary_conditions(
            K,
            FORCES,
            fixed_dofs,
        )
    )

    # --------------------------------------------------------
    # Element results
    # --------------------------------------------------------

    element_results = (
        calculate_element_results(
            U,
            element_data,
            E,
            AREA,
        )
    )

    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {
        "nodes": NODES.copy(),
        "elements": ELEMENTS.copy(),

        "E": E,
        "A": AREA,

        "F": FORCES.copy(),

        "K": K,

        "fixed_dofs": fixed_dofs,
        "free_dofs": free_dofs,

        "U": U,
        "reactions": reactions,

        "element_data": element_data,
        "element_results": element_results,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    result: dict,
) -> None:

    print()
    print("=" * 78)
    print(
        "FEA FROM SCRATCH #03 — 2D TRUSS ANALYSIS"
    )
    print("=" * 78)

    print()
    print("MODEL")
    print("-" * 78)

    print(
        f"Young's modulus : "
        f"{result['E'] / 1e9:.3f} GPa"
    )

    print(
        f"Area            : "
        f"{result['A'] * 1e6:.3f} mm²"
    )

    print(
        "Applied load    : "
        "10.000 kN downward at Node 3"
    )

    print()
    print("NODE COORDINATES")
    print("-" * 78)

    for i, coordinate in enumerate(
        result["nodes"],
        start=1,
    ):

        print(
            f"Node {i}: "
            f"x = {coordinate[0]:.3f} m, "
            f"y = {coordinate[1]:.3f} m"
        )

    print()
    print("GLOBAL DISPLACEMENTS")
    print("-" * 78)

    U = result["U"]

    for i in range(
        len(result["nodes"])
    ):

        ux = U[2 * i]
        uy = U[2 * i + 1]

        print(
            f"Node {i + 1}: "
            f"ux = {ux * 1000: .6f} mm, "
            f"uy = {uy * 1000: .6f} mm"
        )

    print()
    print("REACTIONS")
    print("-" * 78)

    R = result["reactions"]

    for i in range(
        len(result["nodes"])
    ):

        rx = R[2 * i]
        ry = R[2 * i + 1]

        if (
            abs(rx) > 1e-8
            or abs(ry) > 1e-8
        ):

            print(
                f"Node {i + 1}: "
                f"Rx = {rx / 1000: .6f} kN, "
                f"Ry = {ry / 1000: .6f} kN"
            )

    print()
    print("MEMBER RESULTS")
    print("-" * 78)

    for data in result[
        "element_results"
    ]:

        force = (
            data["axial_force"]
            / 1000
        )

        stress = (
            data["stress"]
            / 1e6
        )

        strain = data[
            "strain"
        ]

        length = data[
            "length"
        ]

        if force > 0:
            state = "TENSION"
        elif force < 0:
            state = "COMPRESSION"
        else:
            state = "ZERO FORCE"

        print(
            f"Element {data['element']}: "
            f"L = {length:.6f} m, "
            f"strain = {strain:.6e}, "
            f"stress = {stress:.3f} MPa, "
            f"force = {force:.3f} kN, "
            f"{state}"
        )

    print()
    print("=" * 78)
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = solve_truss()

    print_results(
        result
    )
