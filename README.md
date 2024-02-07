[![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)

# Implementation of LCDNN for solving the General Falkner-Skan equation

This repository contains the implementation of the method which is presented in the following paper:

> [Solving non-linear boundary value problems of Falkner-Skan type equations via Legendre and Chebyshev Neural Blocks](https://doi.org/10.48550/arXiv.2308.03337)

It introduces Legendre and Chebyshev Blocks to solve the general falkner-skan equation.


## Falkner-Skan Equation

    General form:
    
    f''' + α ff'' + β(1 - (f')^2) = 0
    
    with boundary conditions:
    
    f(0) = f'(0) = 0,  and  f'(∞) = 1

## Standard Flows

- **Blasius-Flow**:
    - α = 0.5 , β = 0

- **Pohlhausen-Flow**:
    - α = 0,  β = 1

- **Homann-Flow**:
    - α = 2,  β = 1

- **Hiemenz-Flow**:
    - α = 1,  β = 1

- **Hastings-Flow**:
    - α = 1,  β ∈ [-0.18, 2]

- **Craven-Flow**:
    - α = 1, β ∈ [10, 40]
 
 
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
