"""
Engineering Insight Lab
FEA From Scratch #08 — Complete 2D Truss FEA Solver

A reusable, from-scratch 2D truss finite element solver.

Workflow
--------
1. Define nodes
2. Define truss elements
3. Define material and section properties
4. Assemble the global stiffness matrix
5. Apply nodal loads
6. Apply displacement boundary conditions
7. Partition and solve the reduced system
8. Recover reactions
9. Post-process element strain, stress and axial force
10. Verify equilibrium and matrix symmetry

Global DOF numbering
--------------------
Node i -> [2*i, 2*i+1] in zero-based Python indexing.
For the teaching model:
    [u1, v1, u2, v2, u3, v3]

Sign convention
---------------
Axial force > 0 : tension
Axial force < 0 : compression
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class TrussModel:
    nodes: np.ndarray
    elements: np.ndarray
    E: float
    A: float
    loads: np.ndarray
    fixed_dofs: np.ndarray
    prescribed_values: np.ndarray | None = None


# ============================================================
# GEOMETRY / ELEMENT
# ============================================================

def element_geometry(nodes: np.ndarray, element: np.ndarray):
    """Return i, j, length and direction cosines for one element."""
    i, j = map(int, element)

    xi, yi = nodes[i]
    xj, yj = nodes[j]

    dx = xj - xi
    dy = yj - yi
    L = float(np.hypot(dx, dy))

    if L <= 0.0:
        raise ValueError(
            f"Element {i+1}-{j+1} has zero length."
        )

    c = dx / L
    s = dy / L

    return i, j, L, c, s


def element_stiffness(
    nodes: np.ndarray,
    element: np.ndarray,
    E: float,
    A: float,
):
    """Return the 4x4 global-coordinate stiffness matrix."""
    i, j, L, c, s = element_geometry(nodes, element)

    k = E * A / L

    ke = k * np.array(
        [
            [c*c,  c*s, -c*c, -c*s],
            [c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c,  c*s],
            [-c*s, -s*s, c*s,  s*s],
        ],
        dtype=float,
    )

    dofs = np.array(
        [2*i, 2*i+1, 2*j, 2*j+1],
        dtype=int,
    )

    return ke, dofs, L, c, s


# ============================================================
# GLOBAL ASSEMBLY
# ============================================================

def assemble_global_stiffness(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    A: float,
):
    """Assemble all element stiffness matrices into global K."""
    n_dof = 2 * len(nodes)
    K = np.zeros((n_dof, n_dof), dtype=float)
    element_info = []

    for e, element in enumerate(elements, start=1):
        ke, dofs, L, c, s = element_stiffness(
            nodes,
            element,
            E,
            A,
        )

        for a, I in enumerate(dofs):
            for b, J in enumerate(dofs):
                K[I, J] += ke[a, b]

        element_info.append(
            {
                "element": e,
                "nodes": (
                    int(element[0]) + 1,
                    int(element[1]) + 1,
                ),
                "dofs": dofs,
                "ke": ke,
                "L": L,
                "c": c,
                "s": s,
            }
        )

    return K, element_info


# ============================================================
# LOAD / BOUNDARY-CONDITION HELPERS
# ============================================================

def make_load_vector(
    n_nodes: int,
    nodal_loads: dict[int, tuple[float, float]],
):
    """
    Create global F.

    nodal_loads uses one-based node numbers:
        {3: (0.0, -10000.0)}
    """
    F = np.zeros(2 * n_nodes, dtype=float)

    for node_number, (fx, fy) in nodal_loads.items():
        node = int(node_number) - 1

        if node < 0 or node >= n_nodes:
            raise IndexError(
                f"Node {node_number} is outside the model."
            )

        F[2*node] += fx
        F[2*node + 1] += fy

    return F


def make_fixed_dofs(
    constraints: dict[int, tuple[bool, bool]],
):
    """
    Create constrained DOF indices.

    Example:
        {1: (True, True), 2: (False, True)}

    means:
        Node 1: u1=0, v1=0
        Node 2: v2=0
    """
    fixed = []

    for node_number, (fix_x, fix_y) in constraints.items():
        node = int(node_number) - 1

        if fix_x:
            fixed.append(2*node)

        if fix_y:
            fixed.append(2*node + 1)

    return np.array(sorted(set(fixed)), dtype=int)


# ============================================================
# SOLVER
# ============================================================

def solve_model(model: TrussModel):
    """Solve a general linear 2D truss model."""

    nodes = np.asarray(model.nodes, dtype=float)
    elements = np.asarray(model.elements, dtype=int)

    n_nodes = len(nodes)
    n_dof = 2 * n_nodes

    F = np.asarray(model.loads, dtype=float).reshape(n_dof)

    fixed = np.asarray(
        model.fixed_dofs,
        dtype=int,
    )

    if np.any(fixed < 0) or np.any(fixed >= n_dof):
        raise IndexError("A fixed DOF is outside the global DOF range.")

    all_dofs = np.arange(n_dof)
    free = np.setdiff1d(all_dofs, fixed)

    if model.prescribed_values is None:
        Uc = np.zeros(len(fixed), dtype=float)
    else:
        Uc = np.asarray(
            model.prescribed_values,
            dtype=float,
        ).reshape(len(fixed))

    K, element_info = assemble_global_stiffness(
        nodes,
        elements,
        model.E,
        model.A,
    )

    Kff = K[np.ix_(free, fixed if False else free)]
    Kfc = K[np.ix_(free, fixed)]
    Kcf = K[np.ix_(fixed, free)]
    Kcc = K[np.ix_(fixed, fixed)]

    Ff = F[free]
    Fc = F[fixed]

    rhs = Ff - Kfc @ Uc

    if len(free) == 0:
        Uf = np.zeros(0)
    else:
        Uf = np.linalg.solve(Kff, rhs)

    U = np.zeros(n_dof, dtype=float)
    U[fixed] = Uc
    U[free] = Uf

    # Full-system nodal reactions.
    # R = KU - F
    R = K @ U - F

    # --------------------------------------------------------
    # Element post-processing
    # --------------------------------------------------------

    for item in element_info:
        dofs = item["dofs"]
        c = item["c"]
        s = item["s"]
        L = item["L"]

        ue = U[dofs]

        delta = float(
            np.dot(
                np.array(
                    [-c, -s, c, s],
                    dtype=float,
                ),
                ue,
            )
        )

        strain = delta / L
        stress = model.E * strain
        force = model.A * stress

        item.update(
            {
                "ue": ue,
                "delta": delta,
                "strain": strain,
                "stress": stress,
                "force": force,
                "state": (
                    "TENSION"
                    if force > 0
                    else "COMPRESSION"
                    if force < 0
                    else "ZERO"
                ),
            }
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    symmetry_error = float(
        np.max(np.abs(K - K.T))
    )

    force_balance = K @ U - F - R

    applied_resultant = np.sum(F.reshape(-1, 2), axis=0)
    reaction_resultant = np.sum(
        R.reshape(-1, 2), axis=0,
    )

    global_equilibrium = (
        applied_resultant
        + reaction_resultant
    )

    return {
        "K": K,
        "F": F,
        "U": U,
        "Uf": Uf,
        "R": R,
        "reactions": R,
        "free_dofs": free,
        "fixed_dofs": fixed,
        "Uc": Uc,
        "Kff": Kff,
        "Kfc": Kfc,
        "Kcf": Kcf,
        "Kcc": Kcc,
        "Ff": Ff,
        "Fc": Fc,
        "rhs": rhs,
        "element_info": element_info,
        "symmetry_error": symmetry_error,
        "force_balance_residual": force_balance,
        "applied_resultant": applied_resultant,
        "reaction_resultant": reaction_resultant,
        "global_equilibrium": global_equilibrium,
    }


# ============================================================
# TEACHING EXAMPLE
# ============================================================

def example_model():
    """
    Exact three-member triangular truss used throughout #01–#07.
    """

    nodes = np.array(
        [
            [0.0, 0.0],   # Node 1
            [2.0, 0.0],   # Node 2
            [1.0, 1.5],   # Node 3
        ],
        dtype=float,
    )

    # Zero-based node indices:
    # E1: 1-2
    # E2: 1-3
    # E3: 2-3
    elements = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 2],
        ],
        dtype=int,
    )

    E = 200.0e9
    A = 100.0e-6

    loads = make_load_vector(
        len(nodes),
        {
            3: (0.0, -10_000.0),
        },
    )

    fixed_dofs = make_fixed_dofs(
        {
            1: (True, True),
            2: (False, True),
        }
    )

    return TrussModel(
        nodes=nodes,
        elements=elements,
        E=E,
        A=A,
        loads=loads,
        fixed_dofs=fixed_dofs,
    )


def solve_example():
    return solve_model(example_model())


# ============================================================
# COMMAND-LINE REPORT
# ============================================================

def print_report(result):
    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print("\n==============================================")
    print(" COMPLETE 2D TRUSS FEA SOLVER")
    print("==============================================")

    print("\n--- GLOBAL STIFFNESS MATRIX K [10^6 N/m] ---")
    print(result["K"] / 1.0e6)

    print("\n--- GLOBAL FORCE VECTOR F [N] ---")
    print(result["F"])

    print("\n--- FIXED DOFs ---")
    print(result["fixed_dofs"])

    print("\n--- FREE DOFs ---")
    print(result["free_dofs"])

    print("\n--- REDUCED MATRIX Kff [10^6 N/m] ---")
    print(result["Kff"] / 1.0e6)

    print("\n--- REDUCED FORCE VECTOR Ff [N] ---")
    print(result["Ff"])

    print("\n--- COMPLETE DISPLACEMENT VECTOR [mm] ---")
    print(result["U"] * 1000.0)

    print("\n--- SUPPORT REACTIONS [kN] ---")
    print(result["R"] / 1000.0)

    print("\n--- ELEMENT RESULTS ---")

    for item in result["element_info"]:
        print(
            f"E{item['element']} "
            f"Nodes {item['nodes']}: "
            f"L={item['L']:.6f} m, "
            f"delta={item['delta']*1000:.6f} mm, "
            f"strain={item['strain']:.6e}, "
            f"stress={item['stress']/1e6:.3f} MPa, "
            f"N={item['force']/1000:.3f} kN "
            f"({item['state']})"
        )

    print("\n--- VERIFICATION ---")
    print(
        f"K symmetry error = "
        f"{result['symmetry_error']:.3e}"
    )
    print(
        "Global equilibrium [Fx, Fy] [N] =",
        result["global_equilibrium"],
    )
    print(
        "Force-balance residual max [N] =",
        np.max(np.abs(result["force_balance_residual"])),
    )


if __name__ == "__main__":
    result = solve_example()
    print_report(result)
