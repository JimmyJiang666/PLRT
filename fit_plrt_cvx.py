import numpy as np
import cvxpy as cp
from scipy.sparse import coo_matrix
from itertools import product

def fit_plrt_cvx(X, Y, M, verbose=False):
    Y = Y.reshape(-1)
    n, p = X.shape
    r = np.max(X, axis=0).astype(int)
    F = [[0]]  # Start with predictor index 0 (Python index starts from 0)
    t_n = 700

    for j in range(1, p):  # Loop from second predictor
        flag = 0
        for k in range(len(F)):
            q = F[k][0]
            
            # First CVX optimization problem
            u_j = cp.Variable(r[j])
            u_q = cp.Variable(r[q])

            indices_j = X[:, j].astype(int) - 1
            indices_q = X[:, q].astype(int) - 1

            A_j = coo_matrix((np.ones(n), (np.arange(n), indices_j)), shape=(n, r[j]))
            A_q = coo_matrix((np.ones(n), (np.arange(n), indices_q)), shape=(n, r[q]))

            u_j_terms = A_j.dot(u_j)
            u_q_terms = A_q.dot(u_q)

            total_terms = u_j_terms + u_q_terms

            # **Modification here**
            objective = cp.sum(-cp.multiply(Y, total_terms) + cp.exp(total_terms))

            constraints = [
                u_j >= -2 * np.log(M),
                u_j <= 2 * np.log(M),
                u_q >= -2 * np.log(M),
                u_q <= 2 * np.log(M),
                total_terms >= -np.log(M),
                total_terms <= np.log(M)
            ]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            R_jq_bar = prob.value

            # Second CVX optimization problem
            u_jq = cp.Variable(r[j] * r[q])

            indices_jq = indices_j + indices_q * r[j]
            A_jq = coo_matrix((np.ones(n), (np.arange(n), indices_jq)), shape=(n, r[j] * r[q]))

            u_jq_terms = A_jq.dot(u_jq)

            total_terms = u_jq_terms

            # **Modification here**
            objective = cp.sum(-cp.multiply(Y, total_terms) + cp.exp(total_terms))

            constraints = [
                u_jq >= -np.log(M),
                u_jq <= np.log(M)
            ]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            R_jq = prob.value

            # print(f'G_jq = {R_jq_bar - R_jq}')

            if R_jq_bar - R_jq > t_n:
                F[k].append(j)
                flag = 1
                break

        if flag == 0:
            F.append([j])

    m = len(F)
    rho = np.zeros(m, dtype=int)
    for k in range(m):
        rho[k] = np.prod(r[F[k]])

    cum_rho = np.concatenate(([0], np.cumsum(rho)))
    
    # Build pX as a sparse matrix
    pX_data = []
    pX_row = []
    pX_col = []

    for ind in range(n):
        for k in range(m):
            if len(F[k]) > 1:
                cell_XFk = X[ind, F[k]] - 1  # Adjust for zero-based indexing
                idx = np.ravel_multi_index(cell_XFk, r[F[k]].astype(int))
                pX_row.append(ind)
                pX_col.append(int(cum_rho[k] + idx))
                pX_data.append(1)
            else:
                idx = X[ind, F[k][0]] - 1
                pX_row.append(ind)
                pX_col.append(int(cum_rho[k] + idx))
                pX_data.append(1)
    print("pX coo_matrix building")
    pX = coo_matrix((pX_data, (pX_row, pX_col)), shape=(n, int(cum_rho[m])))

    # Build pA as a sparse matrix
    indices_list = []
    for k in range(m):
        indices = np.arange(int(rho[k]))  # Indices from 0 to rho[k]-1
        indices_list.append(indices)

    all_combinations = list(product(*indices_list))
    total_combinations = len(all_combinations)

    pA_data = []
    pA_row = []
    pA_col = []

    for count, ind_vec in enumerate(all_combinations):
        for k in range(m):
            pA_row.append(count)
            pA_col.append(int(cum_rho[k] + ind_vec[k]))
            pA_data.append(1)
    print("pA coo_matrix building")
    pA = coo_matrix((pA_data, (pA_row, pA_col)), shape=(total_combinations, int(cum_rho[m])))

    # Final CVX optimization problem
    U = cp.Variable(int(cum_rho[m]))
    print("Computing total_terms...")
    total_terms = pX @ U

    # **Modification here**
    objective = cp.sum(-cp.multiply(Y, total_terms) + cp.exp(total_terms))

    constraints = [
        U >= -2 * np.log(M),
        U <= 2 * np.log(M),
        pA @ U >= -np.log(M),
        pA @ U <= np.log(M)
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    if verbose:
        print("Solving CVX optimization with optimized settings...")
        print(f"Number of scalar variables: {prob.size_metrics.num_scalar_variables}")
        print(f"Number of equality constraints: {prob.size_metrics.num_scalar_eq_constr}")
        print(f"Number of inequality constraints: {prob.size_metrics.num_scalar_leq_constr}\n")
    prob.solve(solver=cp.SCS, verbose=False)

    return F, U.value, cum_rho, r, m

import numpy as np
import cvxpy as cp
from scipy.sparse import coo_matrix

def fit_plrt_cvx_optimized(X, Y, M, verbose=False):
    Y = Y.reshape(-1)
    n, p = X.shape
    r = np.max(X, axis=0).astype(int)
    F = [[0]]  # Start with predictor index 0 (Python index starts from 0)
    t_n = 700

    for j in range(1, p):  # Loop from second predictor
        flag = 0
        for k in range(len(F)):
            q = F[k][0]

            # First CVX optimization problem
            u_j = cp.Variable(r[j])
            u_q = cp.Variable(r[q])

            indices_j = X[:, j].astype(int) - 1
            indices_q = X[:, q].astype(int) - 1

            A_j = coo_matrix((np.ones(n), (np.arange(n), indices_j)), shape=(n, r[j]))
            A_q = coo_matrix((np.ones(n), (np.arange(n), indices_q)), shape=(n, r[q]))

            u_j_terms = A_j.dot(u_j)
            u_q_terms = A_q.dot(u_q)

            total_terms = u_j_terms + u_q_terms

            objective = cp.sum(-cp.multiply(Y, total_terms) + cp.exp(total_terms))

            constraints = [
                u_j >= -2 * np.log(M),
                u_j <= 2 * np.log(M),
                u_q >= -2 * np.log(M),
                u_q <= 2 * np.log(M),
                total_terms >= -np.log(M),
                total_terms <= np.log(M)
            ]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            R_jq_bar = prob.value

            # Second CVX optimization problem
            u_jq = cp.Variable(r[j] * r[q])

            indices_jq = indices_j + indices_q * r[j]
            A_jq = coo_matrix((np.ones(n), (np.arange(n), indices_jq)), shape=(n, r[j] * r[q]))

            u_jq_terms = A_jq.dot(u_jq)

            total_terms = u_jq_terms

            objective = cp.sum(-cp.multiply(Y, total_terms) + cp.exp(total_terms))

            constraints = [
                u_jq >= -np.log(M),
                u_jq <= np.log(M)
            ]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            R_jq = prob.value

            if R_jq_bar - R_jq > t_n:
                F[k].append(j)
                flag = 1
                break

        if flag == 0:
            F.append([j])

    m = len(F)
    rho = np.zeros(m, dtype=int)
    for k in range(m):
        rho[k] = np.prod(r[F[k]])

    cum_rho = np.concatenate(([0], np.cumsum(rho)))

    # Initialize used_indices_set
    used_indices_set = set()
    pX_data = []
    pX_row = []
    pX_col = []

    for ind in range(n):
        for k in range(m):
            if len(F[k]) > 1:
                cell_XFk = X[ind, F[k]] - 1  # Adjust for zero-based indexing
                idx = np.ravel_multi_index(cell_XFk, r[F[k]].astype(int))
                col_idx = int(cum_rho[k] + idx)
                pX_row.append(ind)
                pX_col.append(col_idx)
                pX_data.append(1)
                used_indices_set.add(col_idx)
            else:
                idx = X[ind, F[k][0]] - 1
                col_idx = int(cum_rho[k] + idx)
                pX_row.append(ind)
                pX_col.append(col_idx)
                pX_data.append(1)
                used_indices_set.add(col_idx)
    

    used_indices = sorted(list(used_indices_set))
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    pX_col_mapped = [index_mapping[idx] for idx in pX_col]
    print("pX coo_matrix building")
    pX = coo_matrix((pX_data, (pX_row, pX_col_mapped)), shape=(n, len(used_indices)))

    # Final CVX optimization problem
    U = cp.Variable(len(used_indices))
    print("computing total_terms...")
    total_terms = pX @ U

    eta = cp.Variable(m)
    nu = cp.Variable(m)

    log_M = np.log(M)

    # Objective function
    objective = cp.sum(-cp.multiply(Y, total_terms) + cp.exp(total_terms))

    # Constraints
    constraints = []
    constraints += [eta >= -2 * log_M]
    constraints += [nu <= 2 * log_M]
    constraints += [cp.sum(eta) + log_M >= 0]
    constraints += [log_M - cp.sum(nu) >= 0]

    for k in range(m):
        idx_range = []
        for idx in range(int(cum_rho[k]), int(cum_rho[k+1])):
            if idx in used_indices_set:
                mapped_idx = index_mapping[idx]
                idx_range.append(mapped_idx)
        if idx_range:
            constraints += [U[idx_range] >= eta[k]]
            constraints += [U[idx_range] <= nu[k]]
        else:
            pass  # No constraints for this group

    prob = cp.Problem(cp.Minimize(objective), constraints)
    if verbose:
        print("Solving CVX optimization with optimized settings...")
        print(f"Number of scalar variables: {prob.size_metrics.num_scalar_variables}")
        print(f"Number of equality constraints: {prob.size_metrics.num_scalar_eq_constr}")
        print(f"Number of inequality constraints: {prob.size_metrics.num_scalar_leq_constr}\n")
    prob.solve(solver=cp.SCS, verbose=False)

    # Reconstruct the full U vector
    U_full = np.zeros(int(cum_rho[m]))
    for old_idx, new_idx in index_mapping.items():
        U_full[old_idx] = U.value[new_idx]

    return F, U_full, cum_rho, r, m

