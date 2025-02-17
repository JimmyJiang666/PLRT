# PLRT Model Implementation: Code for Low-Rank Approximation and Completion of Positive Tensors

This repository contains Python implementations of the **PLRT** model as described in the paper by Anil Aswani: *"Low-Rank Approximation and Completion of Positive Tensors."* The two files provided, `fit_plrt_cvx.py` and `fit_plrt_mosek.py`, are designed to compute the low-rank representation of positive tensors using convex optimization methods.

## Contents

1. **fit_plrt_cvx.py**: Implements the PLRT model using the CVXPY library with the SCS solver for general-purpose convex optimization.
2. **fit_plrt_mosek.py**: Implements the PLRT model using the MOSEK solver for high-performance optimization.

Both implementations optimize a piecewise log-linear regression model subject to constraints that maintain tensor positivity and log-sum restrictions.

---

## Requirements

### Libraries
Ensure the following Python libraries are installed:
- **NumPy**: For array manipulation.
- **SciPy**: For sparse matrix representations.
- **CVXPY**: For defining and solving convex optimization problems.
- **MOSEK**: (for `fit_plrt_mosek.py`) A licensed solver for large-scale optimization problems.

### For MOSEK, follow installation steps from https://www.mosek.com/downloads/

## Code Description

### 1. **fit_plrt_cvx.py**
#### Function: `fit_plrt_cvx(X, Y, M, verbose=False)`
This implementation uses the **CVXPY** library to solve the PLRT optimization problems.

**Inputs:**
- `X` (array): Predictor matrix with shape `(n, p)`, where `n` is the number of samples and `p` is the number of predictors.
- `Y` (array): Response vector with shape `(n,)`.
- `M` (float): A parameter controlling the magnitude of the constraints.
- `verbose` (bool, optional): If `True`, prints details of the optimization problems.

**Outputs:**
- `F` (list): Grouped predictors identified during the optimization.
- `U` (array): Optimized coefficient values for the PLRT model.
- `cum_rho` (array): Cumulative sum of the rank dimensions for each group.
- `r` (array): Rank dimensions of each predictor.
- `m` (int): Number of groups formed by predictors.

#### Functionality:
- Groups predictors based on log-linear interactions.
- Solves intermediate optimization problems to refine groupings and coefficients.
- Constructs sparse matrices `pX` and `pA` to represent group interactions.
- Optimizes a final objective to compute the PLRT coefficients subject to positivity and log-sum constraints.

---

### 2. **fit_plrt_mosek.py**
#### Function: `fit_plrt_mosek_original(X, Y, r, M, F, verbose=False)`
This implementation uses the **MOSEK** solver for efficient optimization of large-scale problems.

**Inputs:**
- `X` (array): Predictor matrix with shape `(n, p)`.
- `Y` (array): Response vector with shape `(n,)`.
- `r` (array): Maximum rank for each predictor.
- `M` (float): A parameter controlling the magnitude of the constraints.
- `F` (list): Initial grouping of predictors.
- `verbose` (bool, optional): If `True`, prints details of the optimization problems.

**Outputs:**
- `U_value` (array): Optimized coefficients for the PLRT model.
- `pobjval` (float): Final value of the optimization objective.
- `cum_rho` (array): Cumulative sum of rank dimensions for each group.
- `r` (array): Rank dimensions of each predictor.
- `m` (int): Number of groups.

#### Functionality:
- Solves the PLRT problem using the MOSEK solver.
- Constructs sparse matrices for equality and inequality constraints.
- Optimizes a log-linear objective subject to positivity and log-sum constraints.

#### Function: `fit_plrt_mosek_optimized(X, Y, r, M, F, verbose=False)`
This optimized version of the `fit_plrt_mosek_original` function uses advanced precomputations to enhance performance.

---

## Key Differences Between Implementations
1. **fit_plrt_cvx.py**:
   - Relies on the SCS solver within the CVXPY framework.
   - More general-purpose but potentially slower for large-scale problems.

2. **fit_plrt_mosek.py**:
   - Utilizes the MOSEK solver for high-performance optimization.
   - Includes an optimized version for better handling of sparse matrices and precomputations.
