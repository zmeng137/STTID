import os
import tensorly as tl
from tt_cross_lu import TTID_PRRLU_2side, TCI_2site, nested_initIJ_gen_rank1
from tt_svd import TT_SVD
from tensorly.tenalg.proximal import proximal_operator
from tensorly.contrib.decomposition import tensor_train_cross
from utils import TensorSparseStat, read_from_tns

# Quality test for TT-ID (output TCI-format TT, i.e. TT with intermediate pivot matrices)
def TTID_test():
    print("\n===== Testing TT-ID (PRRLU-based) ... =====\n")
    
    TTCores, TTCross_cinv, TTCross_cninv, TTRank = TTID_PRRLU_2side(dense_tensor, rank_max, cutoff)

    reconT = tl.tt_to_tensor(TTCores)
    error = tl.norm(reconT - dense_tensor, 2) / tl.norm(dense_tensor, 2)

    TensorSparseStat(TTCross_cninv)
    print(f"The reconstruction error of STTID is {error}\n")

# Quality test for TT-SVD (with/without LASSO)
def TTSVD_test():
    print("\n===== Testing TT-SVD (with/without LASSO) ... =====\n")
    
    TTCores_svd = TT_SVD(dense_tensor, rank_max, cutoff)
    
    reconT_svd = tl.tt_to_tensor(TTCores_svd)
    error_svd = tl.norm(reconT_svd - dense_tensor, 2) / tl.norm(dense_tensor, 2)

    
    TensorSparseStat(TTCores_svd)
    print(f"The reconstruction error of TT-SVD is {error_svd}\n")

    TTLasso = []
    lasso_thres = 1e-15
    for i in range(len(SHAPE)):
        lasso_core = proximal_operator(TTCores_svd[i], l1_reg=lasso_thres)
        TTLasso.append(lasso_core)

    reconT_lasso = tl.tt_to_tensor(TTLasso)
    error_lasso = tl.norm(reconT_lasso - dense_tensor, 2) / tl.norm(dense_tensor, 2)
    
    TensorSparseStat(TTLasso)
    print(f"The reconstruction error of TT-SVD + LASSO is {error_lasso}\n")

# Quality test for TT-CROSS (PRRLU-based, implemented by us)
def TTcrossLU_test():
    print("\n===== Testing TT-CROSS (PRRLU-based, implemented by us) ... =====\n")

    dim = len(SHAPE)
    interp_I, interp_J = nested_initIJ_gen_rank1(dim, 0)
    
    try:
        TT_cross, TT_cores, TTRank, pr_set, pc_set, result_I, result_J = TCI_2site(dense_tensor, cutoff, rank_max, interp_I, interp_J)
        reconT_tcross = tl.tt_to_tensor(TT_cores)
        error_tcross = tl.norm(reconT_tcross - dense_tensor, 2) / tl.norm(dense_tensor, 2)
        
        TensorSparseStat(TT_cross)
        print(f"The reconstruction error of TT-cross-lu is {error_tcross}\n")

    except Exception as e:
        print(f"Failure in TT-cross-lu: {e}")
        return    
    
# Quality test for TT-CROSS (QR-based, implemented in Tensorly)
def TTcrossQR_test():
    print("\n===== Testing TT-CROSS (QR-based, implemented in Tensorly) ... =====\n")

    random_state = 0
    tol = 1e-2
    maxiter = 10
    
    try:
        tensorly_tt_cross_factors = tensor_train_cross(dense_tensor, rank_list, tol, maxiter, random_state)
        
        reconT_tcross = tl.tt_to_tensor(tensorly_tt_cross_factors)
        error_tcross = tl.norm(reconT_tcross - dense_tensor, 2) / tl.norm(dense_tensor, 2)
        
        TensorSparseStat(tensorly_tt_cross_factors)
        print(f"The reconstruction error of TT-cross-qr is {error_tcross}\n")
        
    except Exception as e:
        print(f"Failure in TT-cross-qr: {e}")
        return
    
# What you need to input for testing
FILENAME = "Rnd1.tns"
SHAPE = [10, 10, 10, 10]
rank_max = 50  # maximal TT-rank for all methods (except for TT-CROSS QR, which uses a different rank parameter)
rank_list = [1, 10, 50, 10, 1]  # rank list for each TT-core (only for TT-CROSS-QR from tensorly, which requires a full rank list instead of a single max rank)
cutoff = 1e-14

# Data Loading
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'Data/random_tensor', FILENAME)
Tensor = read_from_tns(file_path, SHAPE)
dense_tensor = Tensor.todense()

# Tests
TTID_test()       # Our TT-ID
TTSVD_test()      # TT-SVD & TT-SVD-Lasso
TTcrossLU_test()  # TT-CROSS (PRRLU-based, implemented by us)
TTcrossQR_test()  # TT-CROSS (QR-based, implemented in Tensorly)