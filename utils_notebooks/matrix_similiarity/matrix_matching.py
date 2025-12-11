import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# -------------------------------------------------
# Hungarian maximization
# -------------------------------------------------
def hungarian_max(G):
    row_ind, col_ind = linear_sum_assignment(-G)  # maximize = minimize -G
    P = np.zeros_like(G)
    P[row_ind, col_ind] = 1
    return P


# -------------------------------------------------
# Gradient of the relaxed objective
# -------------------------------------------------
def gradient(A, B, S, X):
    return A @ X @ B.T + A.T @ X @ B + S


# -------------------------------------------------
# E/I-aware initialization for X
# -------------------------------------------------
def initialize_X_block_ds(n_E, n_I):
    n = n_E + n_I
    X = np.zeros((n, n))

    X[:n_E, :n_E] = 1.0 / n_E
    X[n_E:, n_E:] = 1.0 / n_I

    return X


# -------------------------------------------------
# Graph matching (simple FW with fixed gamma)
# -------------------------------------------------
def graph_match(A, B, S, n_E, n_I, max_iter=30, tol=1e-5, gamma=0.1):
    n = A.shape[0]
    X = initialize_X_block_ds(n_E, n_I)

    for it in tqdm(range(max_iter)):
        G = gradient(A, B, S, X)
        P = hungarian_max(G)

        X_new = X + gamma * (P - X)

        if np.linalg.norm(X_new - X, "fro") < tol:
            X = X_new
            break

        X = X_new

    P_final = hungarian_max(X)
    return P_final, X


# -------------------------------------------------
# Build the E/I penalty matrix S
# -------------------------------------------------
def EI_similarity_matrix(n_E, n_I, alpha=100.0):
    n = n_E + n_I
    types = np.zeros(n, dtype=int)
    types[n_E:] = 1   # inhibitory group

    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if types[i] != types[j]:
                S[i, j] = -alpha  # strong penalty for E<->I match
    return S

"""
# -------------------------------------------------
# Test with 100×100 sparse matrices
# -------------------------------------------------
np.random.seed(0)

n = 1000
n_E = 800
n_I = 200

density = 0.10  # 10% nonzeros

A = (np.random.rand(n, n) < density).astype(float)
B = (np.random.rand(n, n) < density).astype(float)

S = EI_similarity_matrix(n_E, n_I, alpha=200.0)

P_est, X_est = graph_match(A, B, S, n_E, n_I, max_iter=300, gamma=0.1)

# Frobenius error after matching
B_perm = B[np.ix_(P_est.argmax(axis=1), P_est.argmax(axis=1))]
fro_error = np.linalg.norm(A - B_perm, "fro")

print("Permutation matrix shape:", P_est.shape)
print("Frobenius norm error after matching:", fro_error)
print("First 10 mapped indices:", P_est.argmax(axis=1)[:10])
"""
