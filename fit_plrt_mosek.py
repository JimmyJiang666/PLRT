import numpy as np
import cvxpy as cp
from scipy.sparse import coo_matrix


def fit_plrt_mosek_original(X, Y, r, M, F, verbose=False):
    # Ensure Y is a column vector
    Y = Y.reshape(-1, 1)
    n, p = X.shape
    m = len(F)
    r = np.max(X, axis=0).astype(int)
    
    # Adjust indices in F from MATLAB (1-based) to Python (0-based)
    F = [np.array(f) - 1 for f in F]
    # print(f"r: {r}")
    # Compute rho and cum_rho
    rho = np.zeros(m, dtype=int)
    for k in range(m):
        rho[k] = np.prod(r[F[k]])
    cum_rho = np.hstack(([0], np.cumsum(rho)))
    nvars = cum_rho[-1] + n + 2 * m
    
    # Variables
    U = cp.Variable(cum_rho[-1])
    pX_U = cp.Variable(n)
    eta = cp.Variable(m)
    nu = cp.Variable(m)
    
    # Objective function
    obj = cp.Minimize(cp.sum((1 / n) * cp.exp(pX_U) - cp.multiply(Y.flatten() / n, pX_U)))
    
    # Constraints
    constraints = []
    
    # Lower bounds on eta
    constraints += [eta >= -2 * np.log(M)]
    # Upper bounds on nu
    constraints += [nu <= 2 * np.log(M)]
    
    # Constraints: U >= eta[ind]
    for ind in range(m):
        constraints += [U[cum_rho[ind]:cum_rho[ind+1]] - eta[ind] >= 0]
        constraints += [nu[ind] - U[cum_rho[ind]:cum_rho[ind+1]] >= 0]
    
    # Sum constraints
    constraints += [cp.sum(eta) + np.log(M) >= 0]
    constraints += [np.log(M) - cp.sum(nu) >= 0]
    
    # Equality constraints: pX_U = sum(U at appropriate indices)
    for i in range(n):
        expr = 0
        for k in range(m):
            Fk = F[k]
            if len(Fk) > 1:
                sub_indices = X[i, Fk] - 1  # Adjust for zero-based indexing
                dims = r[Fk]
                idx = np.ravel_multi_index(sub_indices.astype(int), dims.astype(int))
                expr += U[cum_rho[k] + idx]
            else:
                idx = int(X[i, Fk[0]] - 1)
                expr += U[cum_rho[k] + idx]
        constraints += [pX_U[i] == expr]
    prob = cp.Problem(obj, constraints)
    # Solve the problem using MOSEK
    if verbose:
        print("[ORIGINAL] Solving MOSEK optimization...\n")
        print(f"Number of scalar variables: {prob.size_metrics.num_scalar_variables}")
        print(f"Number of equality constraints: {prob.size_metrics.num_scalar_eq_constr}")
        print(f"Number of inequality constraints: {prob.size_metrics.num_scalar_leq_constr}\n")
    prob.solve(solver=cp.MOSEK, verbose=False)
    U_value = U.value
    pobjval = prob.value
    
    return U_value, pobjval, cum_rho, r, m




def fit_plrt_mosek_optimized(X, Y, r, M, F, verbose=False):
    Y = Y.reshape(-1, 1)
    n, p = X.shape
    m = len(F)
    r = np.max(X, axis=0).astype(int)

    # Adjust indices in F from MATLAB (1-based) to Python (0-based)
    F = [np.array(f) - 1 for f in F]

    # Compute rho and cum_rho
    rho = np.array([np.prod(r[f]) for f in F])

    cum_rho = np.hstack(([0], np.cumsum(rho)))


    # Initialize variables
    used_indices_set = set()
    data = []
    row_indices = []
    col_indices = []
    precomputed_indices = []

    # Compute indices sequentially
    for i in range(n):
        expr_list = []
        for k in range(m):
            Fk = F[k]
            if len(Fk) > 1:
                sub_indices = X[i, Fk] - 1  # Adjust for zero-based indexing
                dims = r[Fk]
                idx = np.ravel_multi_index(sub_indices.astype(int), dims.astype(int))
            else:
                idx = int(X[i, Fk[0]] - 1)
            col_idx = cum_rho[k] + idx
            expr_list.append(col_idx)
            used_indices_set.add(col_idx)
            row_indices.append(i)
            col_indices.append(col_idx)
            data.append(1)
        precomputed_indices.append(expr_list)

    # Check if used_indices_set is not empty
    if not used_indices_set:
        raise ValueError("No used indices found. Please check your input data.")

    used_indices = sorted(list(used_indices_set))
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}

    # Define variables
    U = cp.Variable(len(used_indices))
    eta = cp.Variable(m)
    nu = cp.Variable(m)

    # Map old indices to new indices in constraints
    col_indices_mapped = [index_mapping[idx] for idx in col_indices]

    # Build sparse matrix for equality constraints
    A_eq = coo_matrix((data, (row_indices, col_indices_mapped)), shape=(n, len(used_indices)))
    pX_U = A_eq @ U

    # Objective function
    obj = cp.Minimize(cp.sum((1 / n) * cp.exp(pX_U) - cp.multiply(Y.flatten() / n, pX_U)))

    # Constraints
    constraints = []
    log_M = np.log(M)
    constraints += [eta >= -2 * log_M]
    constraints += [nu <= 2 * log_M]

    for ind in range(m):
        idx_range = []
        for idx in range(cum_rho[ind], cum_rho[ind+1]):
            if idx in used_indices_set:
                idx_range.append(index_mapping[idx])
        if idx_range:
            constraints += [U[idx_range] >= eta[ind]]
            constraints += [U[idx_range] <= nu[ind]]
        else:
            # Handle the case where idx_range is empty
            pass  # You may need to adjust your constraints here

    constraints += [cp.sum(eta) + log_M >= 0]
    constraints += [log_M - cp.sum(nu) >= 0]
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    if verbose:
        print("Solving MOSEK optimization with optimized settings...")
        print(f"Number of scalar variables: {prob.size_metrics.num_scalar_variables}")
        print(f"Number of equality constraints: {prob.size_metrics.num_scalar_eq_constr}")
        print(f"Number of inequality constraints: {prob.size_metrics.num_scalar_leq_constr}\n")

    prob.solve(
        solver=cp.MOSEK,
        verbose=False,
        mosek_params={
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-3,
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
        }
    )

    # Reconstruct full U vector
    U_full = np.zeros(cum_rho[-1])
    for old_idx, new_idx in index_mapping.items():
        U_full[old_idx] = U.value[new_idx]

    U_value = U_full
    pobjval = prob.value

    return U_value, pobjval, cum_rho, r, m


