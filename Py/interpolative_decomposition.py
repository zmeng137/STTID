import numpy as np
import time as tm
import utils
from scipy.linalg import solve, qr, eigvals, lu, svd, solve_triangular
from scipy.sparse import random
from typing import Tuple, Union, List

'''
What we have so far...
(i)   interpolative_prrldu
(ii)  interpolative_pqr
(iii) interpolative_sqr
(iv)  interpolative_nuclear
'''

'''========== Interpolative decomposition =========='''
# QR with column pivoting
def PivotedQR(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Array initialization
    Xc = np.copy(X)
    n, p = Xc.shape 
    #t = min(n,p) 
    R = np.zeros([p,p]) # Upper triangular R
    Q = np.zeros([n,p]) # Orthogonal Q
    P = np.arange(p)    # Permutation P
    
    # v_j = ||X[:,j]||^2, j=1,...,n
    v = np.zeros(p)
    for j in range(p):
        v_j = Xc[:,j].T @ Xc[:,j]
        v[j] = v_j
    pk = np.argmax(v)   # Determine an index p1 such that v_p1 is maximal
    maxV = v[pk]
    # Gram-Schmidt process
    rank = 0
    for k in range(p):
        #if k == t:
        #    break
        
        # SWAP X, v, P, R
        Xc[:, [pk, k]] = Xc[:, [k, pk]]
        v[[pk, k]] = v[[k, pk]]
        P[[pk, k]] = P[[k, pk]]
        if k > 0:
            R[0:k,[pk,k]] = R[0:k,[k,pk]]        
        
        # Orthogonalization and R update
        Q[:,k] = Xc[:,k] - Q[:,0:k] @ R[0:k, k]
        R[k,k] = np.sqrt(Q[:,k].T @ Q[:,k])
        Q[:,k] = Q[:,k] / R[k,k]
        # Re-orthogonalization is needed? ...
        R[k,k+1:p] = Q[:,k].T @ Xc[:,k+1:p]
        
        rank += 1 # Rank increment
                
        # Update v_j
        for j in range(k+1, p):
            v[j] = v[j] - R[k,j] * R[k,j]
        # Determine an index p_k+1 >= k+1 such that v_p_k+1 is maximal
        if k < p-1:
            pk = k+1 + np.argmax(v[k+1:])
            pass
        # If v_pk+1 is sufficiently small, leave k
        if v[pk] < 10:
            break
            
    return Q, R, P, rank

# Partial rank-revealing LU
def prrldu(M_: np.ndarray, cutoff: float = 0.0, 
           maxdim: int = np.iinfo(np.int32).max, mindim: int = 1
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], float]:
    """
    Implements the PRRLDU matrix decomposition algorithm (dense v1).
    Args:
        M_: Input matrix
        cutoff: Tolerance for considering values as zero
        maxdim: Maximum dimension for the decomposition
        mindim: Minimum dimension for the decomposition
    Returns:
        Tuple containing (L, d, U, row_perm_inv, col_perm_inv, inf_error)
    """
    assert maxdim > 0, "maxdim must be positive"
    assert mindim > 0, "mindim must be positive"
    mindim = min(maxdim, mindim)
    
    M = M_.copy()
    Nr, Nc = M.shape
    k = min(Nr, Nc)
    
    # Initialize permutations
    rps = list(range(Nr))
    cps = list(range(Nc))
    
    # Find pivots
    inf_error = 0.0
    s = 0
    while s < k:
        Mabs = np.abs(M[s:, s:])
        if Mabs.size == 0:
            break        
        Mabs_max = np.max(Mabs)
        if Mabs_max < cutoff:
            inf_error = Mabs_max
            break
            
        piv = np.unravel_index(np.argmax(Mabs), Mabs.shape)
        piv = (piv[0] + s, piv[1] + s)
        
        # Swap rows and columns
        M[[s, piv[0]], :] = M[[piv[0], s], :]
        M[:, [s, piv[1]]] = M[:, [piv[1], s]]
        
        if s < k - 1:
            M[(s+1):, (s+1):] = M[(s+1):, (s+1):] - np.outer(M[(s+1):, s], M[s, (s+1):]) / M[s, s]
        
        rps[s], rps[piv[0]] = rps[piv[0]], rps[s]
        cps[s], cps[piv[1]] = cps[piv[1]], cps[s]
        s += 1
        #utils.MatrixSparseStat(M)
   
    # Commented on Dec.23/2024
    #M = M_[rps, :][:, cps]
    #utils.MatrixSparseStat(M)
    
    # Initialize L, d, U
    L = np.eye(Nr, k)
    d = np.zeros(k)
    U = np.eye(k, Nc)
    rank = 0
    
    for s in range(min(k, maxdim)):
        P = M[s, s]
        d[s] = P

        if rank < mindim:
            pass
        elif P == 0 or (abs(P) < cutoff and rank + 1 > mindim):
            break
            
        if P == 0:
            P = 1
        rank += 1
        
        if s < Nr - 1:
            piv_col = M[(s+1):, s]
            L[(s+1):, s] = piv_col / P
        if s < Nc - 1:
            piv_row = M[s, (s+1):]
            U[s, (s+1):] = piv_row / P
        
        # Commented on Dec.23/2024
        #if s < k - 1:
        #    M[(s+1):, (s+1):] = M[(s+1):, (s+1):] - np.outer(piv_col, piv_row) / P
        
    L = L[:, :rank]
    d = d[:rank]
    U = U[:rank, :]
    
    # Create inverse permutations
    row_perm_inv = [0] * len(rps)
    for i, p in enumerate(rps):
        row_perm_inv[p] = i
    col_perm_inv = [0] * len(cps)
    for i, p in enumerate(cps):
        col_perm_inv[p] = i
    
    return L, d, U, row_perm_inv, col_perm_inv, inf_error

# Partial rank-revealing LU
def prrldu2(M_: np.ndarray, cutoff: float = 1e-10, 
            maxdim: int = np.iinfo(np.int32).max
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements the PRRLDU matrix decomposition algorithm (dense v2).
    Args:
        M_: Input matrix
        cutoff: Tolerance for rank truncation
        maxdim: Maximum dimension for the decomposition
    Returns:
        Tuple containing (P, L, U)
    """
    # Pivoted LU decomposition (Scipy) 
    assert maxdim > 0, "maxdim must be positive"
    M = M_.copy()
    Nr, Nc = M.shape
    P, L, U = lu(M)    
    
    # Rank truncation
    d = np.abs(U.diagonal())
    r = len(d[d > cutoff])
    r = r if r <= maxdim else maxdim 
    
    # New L, U
    L = L[:, 0:r]
    U = U[0:r, :]    
    return P, L, U

# Interpolative decomposition based on PRRLU
def interpolative_prrldu(M: np.ndarray, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1) -> Tuple[np.ndarray, np.ndarray, List[int], float]:
    """
    Compute interpolative decomposition (ID) from PRRLDU.
    Args:
        M: Input matrix
        **kwargs: Additional keyword arguments passed to prrldu
    Returns:
        Tuple containing (C, Z, pivot_columns, inf_error)
        - C: Matrix containing selected columns
        - Z: Interpolation matrix
        - pivot_columns: List of pivot column indices
        - inf_error: Error measure from PRRLDU
    """
    L, d, U, pr, pc, inf_error = prrldu(M, cutoff, maxdim, mindim)  # Compute PRRLDU decomposition
    k = len(d)
    U11 = U[:, :k]         # Extract relevant submatrices
    iU11 = np.linalg.solve(U11, np.eye(U.shape[0])) # Compute inverse of U11 through backsolving
    ZjJ = iU11 @ U         # Compute interpolation matrix
    CIj = L @ np.diag(d) @ U11   # Compute selected columns
    C = CIj[pr, :]   # Apply row permutation to get C
    Z = ZjJ[:, pc]   # Apply column permutation to get Z
    pivot_cols = [pc.index(i) for i in range(k)]  # Get pivot columns (convert from inverse permutation)
    return C, Z,  pivot_cols, inf_error

# Interpolative decomposition based on PRRLU
def interpolative_prrldu2(M: np.ndarray, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    P, L, U = prrldu2(M, cutoff, maxdim)
    k = L.shape[1]
    U11 = U[:, :k]  # Extract relevant submatrices
    iU11 = np.linalg.solve(U11, np.eye(U.shape[0])) # Compute inverse of U11 through backsolving
    ZjJ = iU11 @ U  # Compute interpolation matrix
    CIj = L @ U11   # Compute selected columns
    C = P @ CIj
    Z = ZjJ
    return C, Z

# Interpolative decomposition based on QRCP
def interpolative_qr(M, maxdim):
    k = maxdim
    #Q , R , P = qr(M, pivoting =True, mode ='economic', check_finite = False)
    Mc = np.copy(M)
    Q , R , P, rank = PivotedQR(Mc)
    if rank < k:
        k = rank
    R_k = R[:k, :k]
    cols = P[:k]
    C = M[:, cols]
    Z = solve(R_k.T @ R_k, C.T @ M, overwrite_a=True, overwrite_b=True, assume_a ='pos')
    approx = C @ Z
    return approx , C , Z
    
def interpolative_sqr(M, maxdim=None):
    row = M.shape[0]
    col = M.shape[1]
    if maxdim is None:
        maxdim = min(M.shape)
    if row <= col:
        K = M @ M.T
    else:
        K = M.T @ M
    svals = eigvals(K) #...TO BE DISCUSSED
    svals = svals[np.sqrt(svals) > 1E-10]
    rank = len(svals)
    maxdim = rank if rank < maxdim else maxdim
    approx, C, Z = interpolative_qr(M, maxdim)
    return approx, C, Z

# Interpolative decomposition by nuclear score
def interpolative_nuclear(M, cutoff=0.0, maxdim=None):
    '''
    Interpolative Decomposition (Nuclear)
    M = C * X
    cutoff - truncation threshold
    maxdim - maximum rank
    '''
    if maxdim is None:
        maxdim = min(M.shape)
    
    maxdim = min(maxdim, M.shape[0], M.shape[1])
    cols = []
    K = M.T @ M
    m = K.shape[0]
    Kt = K
    error = 0.0

    for t in range(maxdim):
        Kt2 = Kt @ Kt
        # Select the column with the maximum score
        l = max((p for p in range(m) if p not in cols), key=lambda p: Kt2[p, p] / Kt[p, p])
        max_err2 = Kt2[l, l] / Kt[l, l]
        cols.append(l)
        error = np.sqrt(np.abs(max_err2))
        if max_err2 < cutoff**2:
            break
        # Schur complement step
        Kt = K - K[:, cols] @ solve(K[np.ix_(cols, cols)], K[cols, :])
    
    # C selection    
    C = M[:, cols]
    
    # X = C \ M
    X = np.zeros([len(cols), M.shape[1]])
    qc, rc = qr(C,mode='economic')
    for i in range(M.shape[1]):
        m = M[:, i]
        z = qc.T @ m
        X[:,i] = solve(rc, z)
     
    # Enforce interpolation structure
    for w in range(len(cols)):
        for r in range(len(cols)):
            X[r, cols[w]] = 1.0 if r == w else 0.0

    return C, X, cols, error

def unit_test_1():
    # Test of interpolative_nuclear
    print("Unit test 1 starts!")
    A = np.array([[3,1],[8,2],[9,-5],[-7,4]])
    B = np.array([[4,6,2],[8,-1,-4]])    
    M = A @ B
    
    maxdim = 2
    cutoff = 1e-4
    C, X, cols, error = interpolative_nuclear(M, cutoff, maxdim)
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 1 ends!")
    return

def unit_test_2():
    # Test of interpolative_sqr 
    print("Unit test 2 starts!")
    m = 10
    n = 8
    rank = 5
    A = np.random.random((m,rank))
    B = np.random.random((rank,n))
    M = A @ B
    
    maxdim = 5
    cutoff = 1e-10
    st = tm.time()
    C, X, cols, error = interpolative_nuclear(M, cutoff, maxdim)
    et = tm.time()
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    print(f"id_nuclear takes {et-st} seconds. The relative recon error = {error}")
    
    st = tm.time()
    approx, C, Z = interpolative_sqr(M, maxdim)
    et = tm.time()
    error = np.linalg.norm(M - approx,ord='fro') / np.linalg.norm(M, ord='fro')    
    print(f"id_sqr takes {et-st} seconds. The relative recon error = {error}")
    print("Unit test 2 ends!")
    return

def unit_test_3():
    # Test of interpolative_prrldu
    print("Unit test 3 starts!")
    m = 12
    n = 11
    rank = 8
    A = np.random.random((m, rank))
    B = np.random.random((rank, n))
    M = A @ B

    cutoff = 1E-5
    maxdim = 9
    C, Z, pivot_cols, inf_error = interpolative_prrldu(M, cutoff, 9)
    error = np.linalg.norm(M - C @ Z, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 3 ends!")
    return

def unit_test_4():
    print("Unit test 4 starts!")
    m = 250
    r = 150
    n = 300
    M = np.random.random((m,r)) @ np.random.random((r,n))
    cutoff = 1E-10

    approx, C, Z = interpolative_qr(M, 150)
    error = np.linalg.norm(M - approx, ord='fro') / np.linalg.norm(M, ord='fro')    
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 4 ends!")
    return

def prrldu_test():
    print("Unit test of partial rank-revealing LDU factorization starts!")
    # Random rank-deficient test matrix
    m = 50
    n = 40
    rank = 30
    min_val = 1
    max_val = 100
    A = np.random.uniform(min_val, max_val, (m,rank))
    B = np.random.uniform(min_val, max_val, (rank,n))
    M = A @ B

    cutoff = 1e-8
    maxdim = 50
    mindim = 1    
    L, d, U, row_perm_inv, col_perm_inv, inf_error = prrldu(M, cutoff, maxdim, mindim)
    
    recon = L @ np.diag(d) @ U
    recon_recover_r = recon[row_perm_inv,:]
    recon_recover_rc = recon_recover_r[:,col_perm_inv]
    max_err = np.max(np.abs(recon_recover_rc - M))    
    print(f"prrldu: revealed rank = {L.shape[1]}, max error = {max_err}")    
    print("Unit test ends!")
    return

def prrldu_sparse_test():
    print("Unit test of partial rank-revealing LDU factorization for a sparse matrix starts!")
    # Random sparse test matrix
    m = 500
    n = 400
    rng = np.random.default_rng()
    M = random(m, n, density=0.1, random_state=rng)
    M_full = M.toarray()
    print("Sparsity statistics of M:")
    utils.MatrixSparseStat(M_full)
    
    cutoff = 1e-8
    maxdim = n
    mindim = 1    
    L, d, U, row_perm_inv, col_perm_inv, inf_error = prrldu(M_full, cutoff, maxdim, mindim)
    
    recon = L @ np.diag(d) @ U
    recon_recover_r = recon[row_perm_inv,:]
    recon_recover_rc = recon_recover_r[:,col_perm_inv]
    max_err = np.max(np.abs(recon_recover_rc - M))    
    print(f"prrldu: revealed rank = {L.shape[1]}, max error = {max_err}")    
    print("Unit test ends!")
    
    return

def pqr_test():
    print("Unit test of pivoted QR factorization starts!")
    # Random rank-deficient test matrix
    m = 50
    n = 40
    rank = 30
    min_val = 0
    max_val = 1000
    A = np.random.uniform(min_val, max_val, (m,rank))
    B = np.random.uniform(min_val, max_val, (rank,n))
    M = A @ B
    
    # Performance of prrldu
    L, d, U, row_perm_inv, col_perm_inv, inf_error = prrldu(M, 1e-7, m, 1)
    recon = L @ np.diag(d) @ U
    recon_recover_r = recon[row_perm_inv,:]
    recon_recover_rc = recon_recover_r[:,col_perm_inv]
    max_err = np.max(np.abs(recon_recover_rc - M))    
    print(f"prrldu: revealed rank = {L.shape[1]}, max error = {max_err}")
    
    # Performance of scipy.qr
    Q1, R1, P1 = qr(M, overwrite_a=False, mode='economic', pivoting=True)
    max_err = np.max(np.abs(Q1 @ R1 - M[:,P1]))
    print(f"scipy.qr: revealed rank = {Q1.shape[1]}, max error = {max_err}")    
    
    # Performance of my pivoted qr
    Q, R, P, rank = PivotedQR(M)
    max_err = np.max(np.abs(Q @ R - M[:,P]))    
    print(f"my pivoted qr: revealed rank = {rank}, max error = {max_err}")    
    print("Unit test ends!")
    
    return

'''========== Unit tests =========='''

#unit_test_1()
#unit_test_2()
#unit_test_3()
#unit_test_4()
#prrldu_test()
#prrldu_sparse_test()
#pqr_test()   # Problem: QR decomposition -> error accumulation? 