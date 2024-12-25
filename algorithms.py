import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular, qr

""" Singular Value Decomposition (SVD) """

def singular_value_decomposition(A, k):
    """
    Compute rank-k approximation of matrix A using truncated SVD.
    
    Parameters:
        A (numpy.ndarray): Input matrix of shape (m, n)
        k (int): Target rank for approximation
        
    Returns:
        U (numpy.ndarray): Left singular vectors (m x k)
        s (numpy.ndarray): Singular values (k)
        Vt (numpy.ndarray): Right singular vectors transposed (k x n)
        A_k (numpy.ndarray): Rank-k approximation of A
    """
    # Compute SVD using numpy library
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Truncate to rank k
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Compute rank-k approximation
    A_k = U_k @ np.diag(s_k) @ Vt_k
    
    return U_k, s_k, Vt_k, A_k



""" Randomised Singular Value Decomposition (Randomised SVD) """

def gram_schmidt_orthonormalize(Y):
    """
    Orthonormalize the columns of Y using the Gram-Schmidt process.
    
    Parameters:
        Y (numpy.ndarray): Input matrix whose columns will be orthonormalized
        
    Returns:
        Q (numpy.ndarray): Matrix Q with orthonormal columns
    """
    if Y.shape[1] == 0:
        return Y
        
    Q = np.zeros_like(Y, dtype=float)  # Explicitly specify float dtype
    
    for i in range(Y.shape[1]):
        # Get the i-th column vector
        q = Y[:, i].copy()  # Make a copy to avoid modifying original
        
        # Subtract projections onto previous vectors
        for j in range(i):
            q -= np.dot(Q[:, j], Y[:, i]) * Q[:, j]
            
        # Normalize the vector
        norm = np.linalg.norm(q)
        if norm > 1e-10:  # Better numerical stability check
            Q[:, i] = q / norm
        else:
            Q[:, i] = np.zeros_like(q)  # Handle linear dependency
            
    return Q

def randomized_svd(A, k, n_oversamples=10, n_power_iterations=1):
    """
    Compute randomized SVD for rank-k approximation of matrix A.
    
    Parameters:
        A (numpy.ndarray): Input matrix of shape (m, n)
        k (int): Target rank
        n_oversamples (int): Additional sampling dimensions for better accuracy
        n_power_iterations (int): Number of power iterations for better accuracy
    
    Returns:
        U (numpy.ndarray): Left singular vectors
        s (numpy.ndarray): Singular values
        Vt (numpy.ndarray): Right singular vectors transposed
        A_k (numpy.ndarray): Rank-k approximation of A
    """
    m, n = A.shape
    p = k + n_oversamples  # Oversampling parameter
    
    # Random sampling
    random_matrix = np.random.normal(0, 1, (n, p))
    Q = A @ random_matrix
    
    # Power iterations to improve accuracy
    for _ in range(n_power_iterations):
        Q = A @ (A.T @ Q)  # Power iteration
        Q = gram_schmidt_orthonormalize(Q)  # Orthogonalize to find matrix Q
    
    # Form smaller matrix B
    B = Q.T @ A
    
    # SVD on the smaller matrix to further lower to exact rank 
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Transform back to original space
    U = Q @ U_tilde
    A_k = U @ np.diag(s) @ Vt # Truncate to desired rank k
    return U[:, :k], s[:k], Vt[:k, :], A_k



""" Rank Revealing QR Decomposition (RRQR) """

def compute_givens(a, b):
    """
    Compute Givens rotation matrix parameters for a 2x2 matrix.
    The rotation matrix G = [[c -s], [s c]] is computed such that G.T @ [a b]^T = [r 0]^T
    
    Parameters:
        a (float): First element
        b (float): Second element
        
    Returns:
        tuple: (c, s) rotation parameters
    """
    if b == 0:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1 / np.sqrt(1 + tau * tau)
        c = s * tau
    else:
        tau = -b / a
        c = 1 / np.sqrt(1 + tau * tau)
        s = c * tau
    return c, s

def rank_revealing_qr_decomposition(A, f, k):
    """
    Strong Rank Revealing QR with fixed rank 'k'
    
    A[:, p] = Q * R = Q [R11, R12; 
                          0, R22]
    where R11 and R12 satisfies that matrix (inv(R11) * R12) has entries
    bounded by a pre-specified constant which should be not less than 1.
    
    Parameters:
        A (ndarray): Target matrix to be approximated
        f (float): Constant that bounds the entries of calculated (inv(R11) * R12)
        k (int): Dimension of R11
        
    Returns:
        tuple: (Q, R, p) where:
            Q (ndarray): m x k orthogonal matrix
            R (ndarray): k x n upper triangular matrix
            p (ndarray): Permutation vector
    """
    # Check constant bound f
    if f < 1:
        print('parameter f given is less than 1. Automatically set f = 2')
        f = 2
    
    # Get matrix dimensions
    m, n = A.shape
    
    # Modify rank k if necessary
    k = min(k, m, n)
    
    # Initial QR factorization with pivoting
    Q, R, p = qr(A, pivoting=True, mode='economic')
    p = p.astype(int)  # Ensure p is integer type
    
    # Check special case
    if k == n:
        print('Rank equals the number of columns!')
        return Q, R, p
    
    # Make diagonals of R positive
    if R.shape[0] == 1 or R.shape[1] == 1:
        ss = np.sign(R[0, 0])
    else:
        ss = np.sign(np.diag(R))
    
    R = R * ss.reshape(-1, 1)
    Q = Q * ss.reshape(1, -1)
    
    # Initialize A^{-1}B (A refers to R11, B refers to R12)
    AB = solve_triangular(R[:k, :k], R[:k, k:], trans=0, unit_diagonal=False)
    
    # Initialize gamma (norm of C's columns, C refers to R22)
    gamma = np.zeros(n - k)
    if k != R.shape[0]:
        gamma = np.sqrt(np.sum(R[k:, k:] ** 2, axis=0))
    
    # Initialize omega (reciprocal of inv(A)'s row norm)
    tmp = solve_triangular(R[:k, :k], np.eye(k), trans=0, unit_diagonal=False)
    omega = 1.0 / np.sqrt(np.sum(tmp ** 2, axis=1))
    
    counter = 0
    while True:
        # Identify interchanging columns
        tmp = np.outer(1.0/omega, gamma) ** 2 + AB ** 2
        ij = np.where(tmp > f * f)
        if len(ij[0]) == 0:
            break
            
        i, j = ij[0][0], ij[1][0]
        counter += 1
        
        # First step: interchange k+1 and k+j th columns
        if j > 0:
            AB[:, [0, j]] = AB[:, [j, 0]]
            gamma[[0, j]] = gamma[[j, 0]]
            R[:, [k, k+j]] = R[:, [k+j, k]]
            p[[k, k+j]] = p[[k+j, k]]
        
        # Second step: interchange i and k th columns
        if i < k:
            idx = list(range(i+1, k+1)) + [i]
            p[i:k+1] = p[idx]
            R[:, i:k+1] = R[:, idx]
            omega[i:k+1] = omega[idx]
            AB[i:k+1, :] = AB[idx, :]
            
            # Givens rotations for triangulation
            for ii in range(i, k):
                c, s = compute_givens(R[ii, ii], R[ii+1, ii])
                G = np.array([[c, -s], [s, c]])
                
                if G[0, 0] * R[ii, ii] + G[0, 1] * R[ii+1, ii] < 0:
                    G = -G
                    
                R[ii:ii+2, :] = G @ R[ii:ii+2, :]
                Q[:, ii:ii+2] = Q[:, ii:ii+2] @ G.T
                
            if R[k, k] < 0:
                R[k, :] = -R[k, :]
                Q[:, k] = -Q[:, k]
        
        # Third step: zeroing out below-diagonal elements
        if k < R.shape[0]:
            for ii in range(k+2, R.shape[0]+1):
                c, s = compute_givens(R[k+1, k+1], R[ii-1, k+1])
                G = np.array([[c, -s], [s, c]])
                
                if G[0, 0] * R[k+1, k+1] + G[0, 1] * R[ii-1, k+1] < 0:
                    G = -G
                    
                R[[k+1, ii-1], :] = G @ R[[k+1, ii-1], :]
                Q[:, [k+1, ii-1]] = Q[:, [k+1, ii-1]] @ G.T
        
        # Fourth step: interchange k and k+1 th columns
        p[k], p[k+1] = p[k+1], p[k]
        ga = R[k, k]
        mu = R[k, k+1] / ga
        nu = R[k+1, k+1] / ga if k < R.shape[0] else 0
        rho = np.sqrt(mu * mu + nu * nu)
        ga_bar = ga * rho
        
        b1 = R[:k, k].copy()
        b2 = R[:k, k+1].copy()
        c1T = R[k, k+2:].copy()
        c2T = R[k+1, k+2:].copy()
        
        c1T_bar = (mu * c1T + nu * c2T) / rho
        c2T_bar = (nu * c1T - mu * c2T) / rho
        
        # Modify R
        R[:k, k] = b2
        R[:k, k+1] = b1
        R[k, k] = ga_bar
        R[k, k+1] = ga * mu / rho
        R[k+1, k+1] = ga * nu / rho
        R[k, k+2:] = c1T_bar
        R[k+1, k+2:] = c2T_bar
        
        # Update AB
        u = solve_triangular(R[:k-1, :k-1], b1[:k-1], trans=0, unit_diagonal=False)
        u1 = AB[:k-1, 0].copy()
        AB[:k-1, 0] = (nu * nu * u - mu * u1) / rho / rho
        AB[k-1, 0] = mu / rho / rho
        AB[k-1, 1:] = c1T_bar / ga_bar
        AB[:k-1, 1:] = AB[:k-1, 1:] + np.outer(nu * u - u1, c2T_bar) / ga_bar
        
        # Update gamma
        gamma[0] = ga * nu / rho
        gamma[1:] = np.sqrt(gamma[1:] ** 2 + c2T_bar ** 2 - c2T ** 2)
        
        # Update omega
        u_bar = u1 + mu * u
        omega[k-1] = ga_bar
        omega[:k-1] = 1.0 / np.sqrt(1.0/omega[:k-1] ** 2 + 
                                   u_bar ** 2 / (ga_bar * ga_bar) - 
                                   u ** 2 / (ga * ga))
        
        # Eliminate new R(k+1, k) by orthogonal transformation
        Gk = np.array([[mu/rho, nu/rho], [nu/rho, -mu/rho]])
        if k < R.shape[0]:
            Q[:, [k, k+1]] = Q[:, [k, k+1]] @ Gk.T
    
    # Recreate low rank approx matrix
    A_approx =  Q[:, :k] @ R[:k, :]
    
    # Return truncated version
    return Q[:, :k], R[:k, :], p, A_approx
