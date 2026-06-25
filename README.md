# FEM & CFD From Scratch — No Black Boxes

Build the **Finite Element Method** and **Computational Fluid Dynamics** from the
ground up, in plain Python — no ANSYS, no COMSOL, no black boxes. Every solver in
this repo is written from scratch and runs in a few lines of NumPy, so you can see
*exactly* what the software you've used has been doing all along.

This is the companion code for the **FEM & CFD From Scratch** video series.
Each episode builds on the last, and the code here grows with it.

> 📺 **Watch the series:** [YouTube playlist link]

---

## Why this exists

Almost every FEM/CFD tutorial online shows you where to click in a commercial
package. You get a colourful stress plot and no idea how it was produced. This
series takes the opposite approach: we derive the method, write the solver, and
verify it against the exact analytical answer — so the understanding is yours,
not the software's.

Everything here is:

- **From scratch** — no FEA/CFD libraries, just NumPy.
- **Verified** — each solver is checked against an analytical solution.
- **Readable** — short, commented, meant to be understood, not just run.

---

## Quick start

```bash
git clone https://github.com/<your-username>/fem-cfd-from-scratch.git
cd fem-cfd-from-scratch
pip install numpy

# run any episode's solver, e.g.:
python episode_01_1d_bar/fem_solver_1d.py
```

That's it — no other dependencies for the solvers.

---

## Episodes

### Phase 1 — The 1D foundation

| # | Episode | Code | Video |
|---|---------|------|-------|
| 1 | The Finite Element Method From Scratch (1D bar) | `episode_01_1d_bar/` | [watch](#) |
| 2 | Shape Functions & Why Convergence Works | `episode_02_shape_functions/` | [watch](#) |
| 3 | The Same Code Now Solves Heat (1D conduction) | `episode_03_heat/` | [watch](#) |

### Phase 2 — Going 2D

| # | Episode | Code | Video |
|---|---------|------|-------|
| 4 | Meshing & Triangular Elements | `episode_04_2d_meshing/` | [watch](#) |
| 5 | The First Real Field (2D heat) | `episode_05_2d_heat/` | _coming soon_ |
| 6 | 2D Stress & Stress Concentration | `episode_06_2d_stress/` | _coming soon_ |

### Phase 3 — Into CFD

| # | Episode | Code | Video |
|---|---------|------|-------|
| 7 | Why CFD Uses Finite Volume | `episode_07_finite_volume/` | _coming soon_ |
| 8 | The Advection–Diffusion Equation | `episode_08_advection_diffusion/` | _coming soon_ |
| 9 | A Simple Navier–Stokes Solver | `episode_09_navier_stokes/` | _coming soon_ |

---

## What's in each episode

**Episode 1 — 1D bar (`fem_solver_1d.py`)**
A fixed–free axial bar under point and distributed loads. Derives the element
stiffness matrix, assembles the global system, applies the boundary condition,
and solves `K u = F`. Verified exact against `δ = PL/EA`; distributed-load case
shows clean O(h²) convergence.

**Episode 3 — 1D heat (`fem_solver_heat_1d.py`)**
The *same* machinery, repurposed for steady heat conduction — conductivity
replaces stiffness, temperature replaces displacement. Handles fixed-temperature
and flux boundary conditions. Verified against Fourier's analytical profile.

_(More solvers are added here as the series progresses.)_

---

## How the code is organised

- `fem_common.py` — shared helpers used across episodes (the solvers, plus the
  Manim animation utilities used to make the videos). If you only want to *run*
  the solvers, you can ignore this and use the standalone `fem_solver_*.py`
  scripts in each episode folder.
- `episode_XX_.../` — the standalone, runnable solver(s) for that episode, kept
  deliberately minimal and self-contained.

The animation source is included too, for anyone curious how the videos are made
— in keeping with the "nothing hidden" spirit of the series.

---

## Running the animations (optional)

The videos are made with [Manim](https://www.manim.community/). If you want to
render the scenes yourself:

```bash
pip install manim
manim -qh episode_01_1d_bar/ep01_scene_03_assembly.py TheAssembly
```

(The solvers do **not** require Manim — only the animations do.)

---

## A note on accuracy

These solvers are built for *clarity*, not production use. They use the simplest
correct formulations (linear elements, basic schemes) so the ideas stay visible.
Real engineering codes add many refinements on top — but every one of those
refinements is a layer on the same foundation built here.

---

## License

MIT — use it, fork it, teach with it. A link back to the series is appreciated
but not required.

---

*Built alongside the FEM & CFD From Scratch series. Questions and corrections
welcome via Issues.*
