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

# For MOSEK, follow installation steps from https://www.mosek.com/downloads/
