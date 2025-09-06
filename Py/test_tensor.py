import os
import sys
import numpy as np
import scipy as sc
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import TTID_PRRLU_2side
from tt_svd import TT_SVD
from tt_cross import tensor_train_cross
from utils import TensorSparseStat, read_from_tns

FILENAME = "powerlaw_test_4d.tns"
SHAPE = [10, 10, 10, 10]
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'Data', FILENAME)
Tensor = read_from_tns(file_path, SHAPE)
dense_tensor = Tensor.todense()

'''
mean = 0
std = 1
nnz = 250
np.random.seed(10)
row_indices = np.random.randint(0, 2500, nnz)
col_indices = np.random.randint(0, 2500, nnz)
values = np.random.normal(mean, std, nnz)
matrix_shape = [2500,2500]    
tensor_shape = [50, 50, 50, 50]
sparse_matrix = sc.sparse.coo_matrix((values, (row_indices, col_indices)), 
                             shape=matrix_shape).tocsr()
    
# Convert to dense, reshape, then back to sparse representation
dense_tensor = sparse_matrix.toarray().reshape(tensor_shape)
'''

rank_max = 50 #max(RANK)
eps = 1e-10
TTCores, TTCross_cinv, TTCross_cninv, TTRank = TTID_PRRLU_2side(dense_tensor, rank_max, eps)

#TTCores_svd = TT_SVD(dense_tensor, rank_max, eps)

reconT = tl.tt_to_tensor(TTCores)
#reconT_svd = tl.tt_to_tensor(TTCores_svd)
error = tl.norm(reconT - dense_tensor, 2) / tl.norm(dense_tensor, 2)
#error_svd = tl.norm(reconT_svd - dense_tensor, 2) / tl.norm(dense_tensor, 2)
print(f"The reconstruction error of TT-ID is {error}")
#print(f"The reconstruction error of TT-SVD is {error_svd}")

print("The sparsity statistics of TT-cores is as follows ...")
TensorSparseStat(TTCores)

print("The sparsity statistics of TTCross_cinv is as follows ...")
TensorSparseStat(TTCross_cinv)

print("The sparsity statistics of TTCross_cninv is as follows ...")
TensorSparseStat(TTCross_cninv)