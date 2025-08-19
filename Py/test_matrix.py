import os
import sys
import numpy as np
import time as tm
from scipy.linalg import qr

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from rank_revealing import prrldu, PivotedQR
from interpolation import interpolative_nuclear, interpolative_prrldu, interpolative_qr, interpolative_sqr
import utils

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

    C, Z, cols = interpolative_qr(M, 200)
    error = np.linalg.norm(M - C @ Z, ord='fro') / np.linalg.norm(M, ord='fro')    
    
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
    L, d, U, row_perm_inv, col_perm_inv, rps, cps, inf_error = prrldu(M, cutoff, maxdim, mindim)
    
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
    L, d, U, row_perm_inv, col_perm_inv, rps, cps, inf_error = prrldu(M, 1e-7, m, 1)
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

unit_test_1()
unit_test_2()
unit_test_3()
unit_test_4()
prrldu_test()
pqr_test()   # Problem: QR decomposition -> error accumulation? 
