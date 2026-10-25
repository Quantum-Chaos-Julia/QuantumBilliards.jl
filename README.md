# QuantumBilliards

[![Build Status](https://github.com/clozej/QuantumBilliards.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/clozej/QuantumBilliards.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Quantum-Chaos-Julia.github.io/QuantumBilliards.jl/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://Quantum-Chaos-Julia.github.io/QuantumBilliards.jl/dev/)
[![Julia](https://img.shields.io/badge/julia-%E2%89%A5%201.8-9558B2.svg?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![SciPost Physics Codebases](https://img.shields.io/badge/SciPost-Codebases-blue.svg)](#citation)
<!-- TODO: replace with real badges after publication:
[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![arXiv](https://img.shields.io/badge/arXiv-PLACEHOLDER-b31b1b.svg)](https://arxiv.org/abs/PLACEHOLDER)
-->

Welcome to the QuantumBilliards.jl - a high performance Julia library for computing the states and spectra of quantum billiards. The goal is to provide a library that is a learning tool for students as well as full computational suite for seasoned quantum chaos researchers that want to attain that last bit of performance for their research publications. We hope this tool will invigorate the study of quantum billiards and related problems and make it more accessible to a wider research community. The documentation is available [here](https://Quantum-Chaos-Julia.github.io/QuantumBilliards.jl/).

## Introduction
Quantum billiards are fundamental models, widely studied in the context of quantum chaos, quantum classical correspondence and semiclassical analysis. 
The package contains a collection of the most effective solution methods for solving quantum billiard eigenvalue problems with state of the art performance and an intuitive API.
In the canonical example we are interested in the quantum equivalent of the mathematical billiard problem where a particle moves freely on the 2D billiard table of a given shape $\mathcal{B} \subset \mathbb{R}^2 $. It is specularly reflected when it hits the boundary of the table. In the quantum case we are interested in the dynamics of a quantum particle in an equivalent setting. We are solving the Schrödinger equation in a 2D infinite potential well of the same shape. The potential can be regarded as $V = 0$ inside the table and $V = \infty$ outside.
To compute the spectrum and eigenstates we must solve the stationary problem, that is the Helmholtz equation

$\left(\nabla^{2}+k_n^{2}\right)\psi_n(x)=0,$

with the boundary condition $\psi_n|_{\partial \mathcal{B}}=0$. Thus we find the eigenenergies $E_n=k_n^2$, where $k_n$ is the wavenumber of the n-th eigenstate.

The library implements extremely efficient numerical methods for computing the spectra (wavenumbers) and states (represented as wavefunctions) allowing us to compute up to a millions of eigenstates!
Each method has its own strengths and limitations and is more suitable for specific use cases for instance either sacrificing some accuracy for computation speed or vice versa. 

# Package features
- General purpose solvers for the 2D interior Dirichlet Helmholtz problem on arbitrary billiard tables: smooth, cornered, symmetry-reduced and multiply connected domains. Geometry is handled by the companion package [BilliardGeometry.jl](https://github.com/Quantum-Chaos-Julia/BilliardGeometry.jl).
- Two complementary formulations behind one interface: Helmholtz **basis methods** (Fourier-Bessel, real plane waves) and **boundary-integral methods** (double-layer and combined-field kernels with Kress corner grading).
- State of the art **accelerated eigensolvers** that return many eigenvalues per step without missing levels: Vergini-Saraceno scaling method, expanded boundary integral method (EBIM), Beyn's contour-integral method and the Compact Rational Krylov (CORK) method. Slow but simple **sweep solvers** are included for validation and basis studies.
- Independent formulations on the same billiard, so spectra can be cross-validated (e.g. with the Weyl law completeness check).
- Large scale computations: complete spectra of $\sim10^5$ levels in a single `compute_spectrum` call (see [Benchmarks](#benchmarks)).
- Eigenstates, wavefunctions, boundary functions (normal derivatives) and Poincaré-Husimi (phase-space) representations, all constructed directly from the returned `SpectralData`.
- Discrete symmetry reduction (reflections, `D2`, `Cn` rotations) with exact, floating-point independent boundary permutations, allowing separation of symmetry sectors and faster computations.
- Highly tunable numerical parameters, quadratures, boundary gradings and basis constructions for fine tuning to specific geometries.
- Multithreaded (`multithreaded=true`) with a type-hierarchy based, multiple-dispatch design: new bases, kernels and solvers plug into existing infrastructure by implementing only the specialized methods.
- Extensible API ready for related problems: magnetic and hyperbolic billiards (in preparation, see [Roadmap](#roadmap)), neutrino billiards, microwave resonators, dielectric microcavities.

# Solvers

| Solver | Formulation | Type | Typical window | Notes |
|---|---|---|---|---|
| `DecompositionMethodSolver`, `ParticularSolutionsMethodSolver` | Helmholtz basis | sweep | single $k$ | Minimizes a tension at trial wavenumbers; levels can be missed between sweep points. Useful for basis convergence studies. |
| `DoubleLayerPotentialSolver` (DLP) | Boundary integral | sweep | single $k$ | Double-layer kernel; may show spurious resonances. |
| `CombinedFieldIntegralEquationSolver` (CFIE), `CompositeBIMSolver` | Boundary integral | sweep | single $k$ | Combined-field kernel avoids spurious resonances; the composite solver allows a different kernel on each boundary component (e.g. annular billiards). |
| `VerginiSaracenoSolver` | Helmholtz basis | accelerated | $\Delta k \approx 0.01$-$0.05$ | Scaling method. Highest throughput; needs a basis that converges for the geometry. |
| `ExpandedBIMSolver` (EBIM) | Boundary integral | accelerated | $\Delta k \approx 0.01$-$0.05$ | Local Taylor expansion of the Fredholm operator with Krylov solves. |
| `BeynSolver` | Boundary integral | accelerated | $\Delta k \approx 0.5$-$1$ | Contour integral in complex $k$-plane, wide windows, robust for any geometry. |
| `CORKSolver` | Boundary integral | accelerated | $\Delta k \approx 0.5$-$1$ | Compact rational Krylov on a Chebyshev/Taylor approximation of the nonlinear problem; lower memory than Beyn. |

All accelerated BIM solvers wrap a sweep kernel (DLP/CFIE), which defines the integral operator and symmetry sector. A detailed cost comparison per window and appendices on parameters are in the paper.

## Choosing a solver
- **Default choice: `BeynSolver` or `CORKSolver`.** They work for general geometries (smooth, cornered, multiply connected) without needing a basis that converges for the shape. Exact performance can depend on the billiard and computer architecture. CORK solver typically has a smaller memory footprint.
- **`VerginiSaracenoSolver` is usefull for star shaped billiards and special shapes where efficient bases are known, such as triangles.** It will be less accurate than the boundary-integral solvers, but it can still be faster and has a much smaller memory footprint. It requires a basis that converges for the geometry (e.g. `CornerAdaptedFourierBessel`), which is not guaranteed for strongly deformed shapes.
- **`ExpandedBIMSolver`** for cross validation.
- **Sweep solvers** only for validation, basis convergence studies and spot checks at single wavenumbers.
- Running two independent solvers on the same billiard is a good way to cross-validate a computed spectrum.

**Bases**: `CornerAdaptedFourierBessel` (corner/singular geometries) and `RealPlaneWaves` (smooth, convex geometries).

**Boundary gradings**: `SmoothPeriodicGrading`, `CornerGrading`, `GlobalCornerGrading` (Kress grading for corner singularities).

# Quick start

```julia
using QuantumBilliards, BilliardGeometry

billiard = C3Billiard(0.2)

kernel = DoubleLayerPotentialSolver(
    8.0,                                  # boundary points per wavelength
    billiard;
    grading = SmoothPeriodicGrading(),
    sym_characters = (1,),                # symmetry sector
)

solver = BeynSolver(kernel; nq = 40, m = 10, Rmax = 0.5,
                    svd_tol = 1e-11, res_tol = 1e-7, eigenvectors = true)

# 8 states starting from the 250-th
k_min, k_max = k_range_for_states(billiard, 250, 257)
spectrum = compute_spectrum(solver, billiard, k_min, k_max; multithreaded = true)

# states (SpectralData is indexable and sliceable)
Psi, x, y = wavefunction(spectrum[1]; b = 5.0, inside_only = true)
u, s      = boundary_function(spectrum[1])
h, p, q   = husimi_function(spectrum[1]; full_p = true)
```

Visualization is provided by [QBPlotting.jl](https://github.com/Quantum-Chaos-Julia/QBPlotting.jl) (Makie based).

# Billiards
Available through BilliardGeometry.jl: circle, ellipse, rectangle, stadium, Prosen, Robnik limaçon, mushroom, annular, Sinai, triangles, general polygons, polar/Fourier-ripple billiards, `C3` and star billiards, and more. Custom billiards are assembled from parametrized curves (`LineSegment`, `CircleSegment`, `PolarSegment`, `FourierCoeffPolarSegment`), grouped into simple or composite domains with boundary roles (specular reflection, transparent interface, symmetry wall) and an optional symmetry registry.

# Benchmarks
Complete spectra for $500 \le k \le 1400$ of the Prosen billiard (single `compute_spectrum` call, 64 threads, AMD Threadripper PRO 7985WX, Julia 1.12 with MKL):

| Solver | Eigenvalues | Runtime [s] | Eigenvalues/s |
|---|---|---|---|
| `VerginiSaracenoSolver` | 106 697 | 2875 | 37.1 |
| `BeynSolver` | 108 710 | 3338 | 32.6 |
| `CORKSolver` | 108 710 | 4125 | 26.4 |

(VS uses `a=0.04`, BIM solvers `a=0.2`, so level counts differ.) Completeness was verified against the Weyl law.

# Related packages
- [BilliardGeometry.jl](https://github.com/Quantum-Chaos-Julia/BilliardGeometry.jl) - geometry, boundary discretization, quadratures, symmetries.
- [QBPlotting.jl](https://github.com/Quantum-Chaos-Julia/QBPlotting.jl) - plotting of states, boundary functions and Husimi functions.
- [SpectralStatistics.jl](https://github.com/Quantum-Chaos-Julia/SpectralStatistics.jl) - spectral statistics and random matrix models for the computed spectra.
- `ClassicalBilliards.jl` - classical billiard dynamics (under development).

# Roadmap
- Billiards on the hyperbolic plane and in a constant magnetic field (Beyn and CORK).
- Neutrino billiards, dielectric microcavities, time evolution of billiard states.
- Recursively compressed inverse preconditioning, corner charge bases, QBX/QB2X quadratures, Neumann-to-Dirichlet spectral flow.

# Citation
If you use QuantumBilliards.jl in your research, please cite:

```bibtex
@article{OrelLozej2026QuantumBilliards,
  title   = {QuantumBilliards.jl: An extensible, user friendly, high-performance Julia library for the large scale computation of spectra and eigenstates of quantum billiards and related problems},
  author  = {Orel, Matic and Lozej, {\v C}rt},
  journal = {SciPost Physics Codebases},
  year    = {2026},
  note    = {PLACEHOLDER: add volume, article number, DOI and arXiv identifier after publication}
}
```

# License
See [LICENSE](LICENSE).