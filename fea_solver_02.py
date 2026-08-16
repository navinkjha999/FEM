"""
fea_solver_02.py

Engineering From Scratch — Finite Element Analysis
Tutorial 02: Mesh Refinement & Convergence

Pure numerical solver.
No Manim dependency.

Problem
-------
Uniform axial bar:

    Fixed                              F = 10 kN
      |====================================>
      |<------------- L = 2 m ------------>|

    A = 100 mm²
    E = 200 GPa

The same physical problem is solved using:

    1, 2, 4, 8, 16 and 32 elements.

The purpose is to investigate mesh refinement and convergence.

Important:
----------
For this particular problem, the exact displacement field is linear.
A linear 1D axial-bar finite element can therefore represent the exact
solution even with one element.

Consequently, the tip displacement remains essentially:

    1.0 mm

for every mesh.

This is intentional and is explained in the Manim video.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Problem definition
# ============================================================

TOTAL_LENGTH = 2.0          # m
AREA = 100e-6               # m² = 100 mm²
YOUNGS_MODULUS = 200e9      # Pa = 200 GPa
APPLIED_FORCE = 10e3        # N = 10 kN

ELEMENT_COUNTS = [
    1,
    2,
    4,
    8,
    16,
    32,
]


# ============================================================
# Element stiffness
# ============================================================

def axial_bar_stiffness(
    E: float,
    A: float,
    length: float,
) -> np.ndarray:
    """
    Linear 2-node axial bar element stiffness matrix.

        [ke] = AE/L * [[ 1, -1],
                       [-1,  1]]
    """

    if E <= 0:
        raise ValueError(
            "Young's modulus must be positive."
        )

    if A <= 0:
        raise ValueError(
            "Cross-sectional area must be positive."
        )

    if length <= 0:
        raise ValueError(
            "Element length must be positive."
        )

    return (
        E * A / length
        * np.array(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
            ]
        )
    )


# ============================================================
# Assemble global stiffness matrix
# ============================================================

def assemble_global_stiffness(
    E: float,
    A: float,
    total_length: float,
    number_of_elements: int,
) -> tuple[np.ndarray, list[np.ndarray]]:

    if number_of_elements < 1:
        raise ValueError(
            "Number of elements must be at least 1."
        )

    number_of_nodes = number_of_elements + 1
    element_length = (
        total_length / number_of_elements
    )

    K = np.zeros(
        (number_of_nodes, number_of_nodes)
    )

    element_matrices = []

    for element in range(number_of_elements):

        ke = axial_bar_stiffness(
            E=E,
            A=A,
            length=element_length,
        )

        element_matrices.append(ke)

        # Global node indices.
        node_i = element
        node_j = element + 1

        # Assembly.
        K[node_i, node_i] += ke[0, 0]
        K[node_i, node_j] += ke[0, 1]
        K[node_j, node_i] += ke[1, 0]
        K[node_j, node_j] += ke[1, 1]

    return K, element_matrices


# ============================================================
# Solve KU = F with prescribed displacement
# ============================================================

def solve_system(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: list[int],
) -> tuple[np.ndarray, np.ndarray]:

    number_of_dofs = len(F)
    all_dofs = np.arange(number_of_dofs)

    free_dofs = np.array(
        [
            dof
            for dof in all_dofs
            if dof not in fixed_dofs
        ],
        dtype=int,
    )

    # Reduced system.
    K_ff = K[
        np.ix_(free_dofs, free_dofs)
    ]

    F_f = F[free_dofs]

    # Unknown displacement vector.
    U = np.zeros(number_of_dofs)

    U[free_dofs] = np.linalg.solve(
        K_ff,
        F_f,
    )

    # Reactions.
    reactions = K @ U - F

    return U, reactions


# ============================================================
# Element post-processing
# ============================================================

def calculate_element_results(
    U: np.ndarray,
    E: float,
    A: float,
    total_length: float,
    number_of_elements: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    element_length = (
        total_length / number_of_elements
    )

    strains = []
    stresses = []
    internal_forces = []

    for element in range(number_of_elements):

        node_i = element
        node_j = element + 1

        u_i = U[node_i]
        u_j = U[node_j]

        strain = (
            u_j - u_i
        ) / element_length

        stress = E * strain
        internal_force = A * stress

        strains.append(strain)
        stresses.append(stress)
        internal_forces.append(
            internal_force
        )

    return (
        np.array(strains),
        np.array(stresses),
        np.array(internal_forces),
    )


# ============================================================
# Analytical solution
# ============================================================

def analytical_solution(
    E: float,
    A: float,
    total_length: float,
    force: float,
) -> tuple[float, float]:

    displacement = (
        force * total_length
        / (A * E)
    )

    stress = force / A

    return displacement, stress


# ============================================================
# Solve one mesh
# ============================================================

def solve_one_mesh(
    number_of_elements: int,
    total_length: float = TOTAL_LENGTH,
    area: float = AREA,
    youngs_modulus: float = YOUNGS_MODULUS,
    applied_force: float = APPLIED_FORCE,
) -> dict:

    number_of_nodes = (
        number_of_elements + 1
    )

    element_length = (
        total_length / number_of_elements
    )

    # Global stiffness.
    K, element_matrices = (
        assemble_global_stiffness(
            E=youngs_modulus,
            A=area,
            total_length=total_length,
            number_of_elements=number_of_elements,
        )
    )

    # Global force vector.
    F = np.zeros(number_of_nodes)

    F[-1] = applied_force

    # Node 1 fixed.
    fixed_dofs = [0]

    # Solve.
    U, reactions = solve_system(
        K=K,
        F=F,
        fixed_dofs=fixed_dofs,
    )

    # Element results.
    strains, stresses, internal_forces = (
        calculate_element_results(
            U=U,
            E=youngs_modulus,
            A=area,
            total_length=total_length,
            number_of_elements=number_of_elements,
        )
    )

    # Exact solution.
    exact_displacement, exact_stress = (
        analytical_solution(
            E=youngs_modulus,
            A=area,
            total_length=total_length,
            force=applied_force,
        )
    )

    tip_displacement = U[-1]

    # Relative displacement error.
    displacement_error = (
        abs(
            tip_displacement
            - exact_displacement
        )
        / abs(exact_displacement)
        * 100.0
    )

    return {
        "number_of_elements": number_of_elements,
        "number_of_nodes": number_of_nodes,
        "element_length": element_length,

        "K": K,
        "F": F,
        "element_matrices": element_matrices,

        "U": U,
        "reactions": reactions,

        "strains": strains,
        "stresses": stresses,
        "internal_forces": internal_forces,

        "tip_displacement": tip_displacement,

        "exact_displacement":
            exact_displacement,

        "exact_stress":
            exact_stress,

        "displacement_error_percent":
            displacement_error,
    }


# ============================================================
# Complete convergence study
# ============================================================

def run_convergence_study(
    element_counts: list[int] | None = None,
) -> dict:

    if element_counts is None:
        element_counts = ELEMENT_COUNTS

    cases = []

    for number_of_elements in element_counts:

        result = solve_one_mesh(
            number_of_elements=
                number_of_elements
        )

        cases.append(result)

    # Exact solution is common to all cases.
    exact_displacement = cases[0][
        "exact_displacement"
    ]

    exact_stress = cases[0][
        "exact_stress"
    ]

    return {
        "total_length": TOTAL_LENGTH,
        "area": AREA,
        "youngs_modulus":
            YOUNGS_MODULUS,
        "applied_force":
            APPLIED_FORCE,

        "element_counts":
            list(element_counts),

        "exact_displacement":
            exact_displacement,

        "exact_stress":
            exact_stress,

        "cases": cases,
    }


# ============================================================
# Print convergence table
# ============================================================

def print_convergence_table(
    study: dict,
) -> None:

    print()
    print("=" * 78)
    print(
        "FEA FROM SCRATCH #02 — "
        "MESH REFINEMENT & CONVERGENCE"
    )
    print("=" * 78)

    print()
    print("PHYSICAL PROBLEM")
    print("-" * 78)

    print(
        f"Length            = "
        f"{study['total_length']:.3f} m"
    )

    print(
        f"Area              = "
        f"{study['area']:.6e} m²"
    )

    print(
        f"Young's modulus   = "
        f"{study['youngs_modulus']:.6e} Pa"
    )

    print(
        f"Applied force     = "
        f"{study['applied_force']:.3f} N"
    )

    print()
    print("ANALYTICAL SOLUTION")
    print("-" * 78)

    print(
        f"Tip displacement  = "
        f"{study['exact_displacement'] * 1000:.6f} mm"
    )

    print(
        f"Stress            = "
        f"{study['exact_stress'] / 1e6:.6f} MPa"
    )

    print()
    print("MESH CONVERGENCE")
    print("-" * 78)

    print(
        f"{'Elements':>10}"
        f"{'Nodes':>10}"
        f"{'Le (m)':>12}"
        f"{'Tip disp. (mm)':>18}"
        f"{'Error (%)':>15}"
    )

    print("-" * 78)

    for case in study["cases"]:

        print(
            f"{case['number_of_elements']:>10}"
            f"{case['number_of_nodes']:>10}"
            f"{case['element_length']:>12.5f}"
            f"{case['tip_displacement'] * 1000:>18.6f}"
            f"{case['displacement_error_percent']:>15.6e}"
        )

    print("-" * 78)

    print()
    print(
        "Observation: for this uniform bar with linear "
        "axial elements, the exact linear displacement "
        "field is represented even with one element."
    )

    print("=" * 78)
    print()


# ============================================================
# Standalone execution
# ============================================================

if __name__ == "__main__":

    study = run_convergence_study()

    print_convergence_table(study)
