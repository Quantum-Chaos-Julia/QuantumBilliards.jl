# Home
Welcome to the QuantumBilliards.jl - a high performance Julia library for computing the states and spectra of quantum billiards. The goal is to provide a library that is a learning tool for students as well as full computational suite for seasoned quantum chaos researchers that want to attain that last bit of performance for their research publications. We hope this tool will invigorate the study of quantum billiards and related problems and make it more accessible to a wider research community. 

## Introduction
Quantum billiards are fundamental models, widely studied in the context of quantum chaos, quantum classical correspondence and semiclassical analysis. 
The package contains a collection of the most effective solution methods for solving quantum billiard eigenvalue problems with state of the art performance and an intuitive API.
In the canonical example we are interested in the quantum equivalent of the mathematical billiard problem where a particle moves freely on the 2D billiard table of a given shape $\mathcal{B} \subset \mathbb{R}^2 $. It is specularly reflected when it hits the boundary of the table. In the quantum case we are interested in the dynamics of a quantum particle in an equivalent setting. We are solving the Schrödinger equation in a 2D infinite potential well of the same shape. The potential can be regarded as $V = 0$ inside the table and $$V = \infty$$ outside.
To compute the spectrum and eigenstates we must solve the stationary problem, that is the Helmholtz equation

$\left(\nabla^{2}+k_n^{2}\right)\psi_n(x)=0,$

with the boundary condition $\psi_n|_{\partial \mathcal{B}}=0$. Thus we find the eigenenergies $E_n=k_n^2$, where $k_n$ is the wavenumber of the n-th eigenstate.

The library implements extremely efficient numerical methods for computing the spectra (wavenumbers) and states (represented as wavefunctions) allowing us to compute up to a millions of eigenstates!
Each method has its own strengths and limitations and is more suitable for specific use cases for instance either sacrificing some accuracy for computation speed or vice versa. 

The Tutorial section explains how to get started using QuantumBilliards.

See the API section for the complete documentation on all the exported types and functions. 
## Package features
- General purpose solvers that can handle arbitrary billiard tables - seamless integration with BilliardGeometry.jl
- State of the art billiard solvers featuring all the most efficient solution methods: Vergini-Saraceno scaling, expanded BIM, Beyn's contour method and CORK.
- Eigenstates, wavefunctions, boundary functions, and Husimi (phase-space) representations.
- Highly tunable numerical parameters, integration schemes and basis constructions for fine tuning to specific geometries.
- Discrete symmetry reduction schemes, allows separation of spectral symmetry sectors and improved computation efficiency.  
- Computation planning: window layout, expected number of states and an upper-bound memory estimate before any eigen-solve (`plan_computation`).
- Extensible API ready for future updates to related problems like microwave resonators, neutrino billiards etc.

## Contents
```@contents
```
