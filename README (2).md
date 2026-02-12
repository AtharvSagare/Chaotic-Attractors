# Chaos Attractors: Numerical Simulation and 3D Visualization

## Abstract

This repository presents numerical simulations and real-time 3D
visualizations of classical chaotic dynamical systems. The project
focuses on deterministic nonlinear systems that exhibit sensitive
dependence on initial conditions, a fundamental property of chaos
theory.

All systems are integrated using the Fourth-Order Runge--Kutta (RK4)
method and rendered using GPU-based visualization techniques via
`vispy`.

The objective of this repository is to combine mathematical modeling,
numerical analysis, computational physics, and scientific visualization
within an open-source and extensible framework.

------------------------------------------------------------------------

## 1. Introduction to Chaos Theory

Chaos theory studies deterministic nonlinear dynamical systems whose
long-term behavior appears unpredictable due to extreme sensitivity to
initial conditions.

Even though these systems follow precise mathematical laws, they
demonstrate:

-   Sensitive dependence on initial conditions (Butterfly Effect)
-   Aperiodic long-term evolution
-   Strange attractors in phase space
-   Complex geometric structures emerging from simple equations

Mathematically, chaotic systems are described by nonlinear ordinary
differential equations (ODEs):

dx/dt = F(x)

Small perturbations in initial conditions can lead to exponentially
diverging trajectories.

This repository explores three classical chaotic systems:

-   Lorenz Attractor\
-   Rössler Attractor\
-   Halvorsen Attractor

------------------------------------------------------------------------

## 2. Numerical Method

All dynamical systems in this repository are integrated using the:

### Fourth-Order Runge--Kutta Method (RK4)

RK4 provides a high-accuracy numerical solution for first-order ODE
systems:

x(n+1) = x(n) + dt/6 \* (k1 + 2k2 + 2k3 + k4)

This method balances computational efficiency and numerical stability.

Contributors are encouraged to experiment with:

-   Euler Method\
-   RK2 / Midpoint Method\
-   Adaptive step solvers\
-   Higher-order integrators

Comparative analysis of numerical stability and attractor structure
would be a valuable extension.

------------------------------------------------------------------------

## 3. Repository Structure

### 3.1 Lorenz Attractor

File: `lorentz smooth.py`

The Lorenz system is defined as:

dx/dt = sigma (y - x)\
dy/dt = x (rho - z) - y\
dz/dt = xy - beta z

This implementation provides a smooth formation of the classical Lorenz
strange attractor using RK4 integration and GPU rendering.

This file serves as the foundational and most fundamental implementation
within the repository.

------------------------------------------------------------------------

### 3.2 Rössler Attractor (Interactive Visualization)

File: `ROSSLERZOOM.py`

The Rössler system is defined as:

dx/dt = -y - z\
dy/dt = x + a y\
dz/dt = b + z (x - c)

Features:

-   Progressive attractor formation\
-   Interactive rotation\
-   Zoom capability\
-   Shader-based glow visualization\
-   Background music integration

To enable the complete experience:

1.  Download `quantum.mp3` (included in this repository).\
2.  Place it in the same directory as `ROSSLERZOOM.py`.\
3.  Run the script.

The background track played is:

"Quantum Mechanics -- Oppenheimer"

------------------------------------------------------------------------

### 3.3 Halvorsen Attractor

File: `halvorsen.py`

The Halvorsen system is another nonlinear chaotic attractor implemented
using RK4 integration.

Features:

-   Long trajectory integration\
-   Constant-speed visualization build-up\
-   Interactive controls\
-   Background music support (Spacebar to pause/resume)

------------------------------------------------------------------------

## 4. Installation and Execution

### 4.1 Dependencies

Install required packages:

pip install numpy vispy pygame

### 4.2 Running the Programs

Run any file individually:

python "lorentz smooth.py"\
python ROSSLERZOOM.py\
python halvorsen.py

Ensure that `quantum.mp3` is placed in the same folder when running
files that require audio playback.

------------------------------------------------------------------------

## 5. Open-Source Contribution

This repository is released under the MIT License.

Users are encouraged to:

-   Modify and improve numerical methods\
-   Add new chaotic attractors\
-   Optimize rendering performance\
-   Improve documentation\
-   Implement real-time parameter control\
-   Conduct numerical stability comparisons

The objective is to gradually expand this repository into a
comprehensive computational visualization framework for nonlinear
dynamical systems.

------------------------------------------------------------------------

## 6. Future Work

Planned extensions include:

-   Additional strange attractors\
-   Parameter bifurcation studies\
-   Lyapunov exponent computation\
-   Real-time interactive parameter sliders\
-   Comparative integrator analysis\
-   Improved shader and rendering techniques

------------------------------------------------------------------------

## 7. Conclusion

Chaotic systems demonstrate how deterministic equations can generate
extraordinary complexity.

This repository aims to bridge:

-   Nonlinear dynamics\
-   Numerical analysis\
-   Scientific computing\
-   Visual simulation\
-   Computational aesthetics

Contributions and collaborative development are strongly welcomed.
