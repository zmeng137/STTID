import os
import sys
import numpy as np
import scipy as sc
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import TTID_PRRLU_2side, TCI_2site, nested_initIJ_gen_rank1
from tt_svd import TT_SVD
from tensorly.decomposition import tensor_train as TT_SVD_tl
from tensorly.tenalg.proximal import proximal_operator
from tensorly.contrib.decomposition import tensor_train_cross
from utils import TensorSparseStat, read_from_tns

def TCI_test():
    dim = len(SHAPE)
    interp_I, interp_J = nested_initIJ_gen_rank1(dim, 0)
    TT_cross, TT_cores, TTRank, pr_set, pc_set, result_I, result_J = TCI_2site(dense_tensor, eps, rank_max, interp_I, interp_J)
    
    reconT_tcross = tl.tt_to_tensor(TT_cores)
    error_tcross = tl.norm(reconT_tcross - dense_tensor, 2) / tl.norm(dense_tensor, 2)
    
    print(f"The reconstruction error of TCI is {error_tcross}")
    TensorSparseStat(TT_cross)

    return

def TTcross_test():
    random_state = 0
    tol = 1e-2
    maxiter = 10
    rank = [1, 50, rank_max, 50, 1]
    
    tensorly_tt_cross_factors = tensor_train_cross(dense_tensor, rank, tol, maxiter, random_state)
    
    reconT_tcross = tl.tt_to_tensor(tensorly_tt_cross_factors)
    error_tcross = tl.norm(reconT_tcross - dense_tensor, 2) / tl.norm(dense_tensor, 2)
    
    print(f"The reconstruction error of TT-CROSS is {error_tcross}")
    TensorSparseStat(tensorly_tt_cross_factors)

    return

def TTSVD_test():
    #TTCores_svd = TT_SVD(dense_tensor, rank_max, eps)
    #TTCores_svd = TT_SVD_tl(dense_tensor, rank_max, 'symeig_svd')
    TTCores_svd = TT_SVD_tl(dense_tensor, rank_max)

    reconT_svd = tl.tt_to_tensor(TTCores_svd)
    error_svd = tl.norm(reconT_svd - dense_tensor, 2) / tl.norm(dense_tensor, 2)
    
    print(f"The reconstruction error of TT-SVD is {error_svd}")
    print("The sparsity statistics of TT-SVD is as follows ...")
    TensorSparseStat(TTCores_svd)

    return

def STTID_test():
    TTCores, TTCross_cinv, TTCross_cninv, TTRank = TTID_PRRLU_2side(dense_tensor, rank_max, eps)

    reconT = tl.tt_to_tensor(TTCores)
    error = tl.norm(reconT - dense_tensor, 2) / tl.norm(dense_tensor, 2)

    print(f"The reconstruction error of STTID is {error}")
    print("The sparsity statistics of STTID_crossninv is as follows ...")
    TensorSparseStat(TTCross_cninv)
    
    return

def TTLasso_test():
    TTCores_svd = TT_SVD(dense_tensor, rank_max, eps)
    
    reconT_svd = tl.tt_to_tensor(TTCores_svd)
    error_svd = tl.norm(reconT_svd - dense_tensor, 2) / tl.norm(dense_tensor, 2)
    
    print(f"The reconstruction error of TT-SVD is {error_svd}")
    print("The sparsity statistics of TT-SVD is as follows ...")
    TensorSparseStat(TTCores_svd)

    TTLasso = []
    lasso_thres = 1e-15
    for i in range(len(SHAPE)):
        lasso_core = proximal_operator(TTCores_svd[i], l1_reg=lasso_thres)
        TTLasso.append(lasso_core)

    reconT_lasso = tl.tt_to_tensor(TTLasso)
    error_lasso = tl.norm(reconT_lasso - dense_tensor, 2) / tl.norm(dense_tensor, 2)
    print(f"The reconstruction error of TT-SVD + LASSO is {error_lasso}")
    print("The sparsity statistics of TT-SVD + LASSO is as follows ...")
    TensorSparseStat(TTLasso)

    return

FILENAME = "rnd_pl_8.tns"
SHAPE = [100, 100, 100, 100]
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'Data', FILENAME)
Tensor = read_from_tns(file_path, SHAPE)
dense_tensor = Tensor.todense()

rank_max = 40000 #max(RANK)
eps = 1e-14

#TTLasso_test()
#TTcross_test()
TTSVD_test()