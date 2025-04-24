import numpy as np
import os

import tensorly as tl
from tensorly.contrib.decomposition import tensor_train_cross
from tt_svd import TT_SVD
from tt_id import TT_IDscatter, TT_IDPRRLDU, TT_IDPRRLDU_Forward
from utils import TensorSparseStat, read_from_tns

def testCase1():
    print("Unit test 1 starts!")
    
    rank = [1, 3, 3, 1]        # TT rank
    order = [5, 5, 5]         # tensor order
    density = [1, 1, 1]    # density for every factor
    seed = [1, 2, 3]             # random seeds
    factors = []                 # factor list 

    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        np.random.seed(seed[i])
        factor = np.random.random(shape)
        randl = np.random.random(shape)
        factor = np.where(randl>density[i], 0, factor)
        factors.append(factor)                
    SpTd = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
    TensorSparseStat([SpTd])
    
    '''
    outputFlag = 0
    if outputFlag == 1:     
        tnsName = "syn_order_" + "_".join(map(str, order)) + "_synrank_" + "_".join(map(str, rank)) + ".tns"
        tnsPath = "/home/mengzn/Desktop/TensorData/" + tnsName
        cntData = SpT.nnz
        nnzData = SpT.data
        coord = SpT.coords
        with open(tnsPath, "w") as f:
            for i in range(cntData):
                f.write(f"{coord[0][i]} {coord[1][i]} {coord[2][i]} {nnzData[i]}\n")
    '''

    # TT-SVD
    rank_max = max(rank)
    eps = 1e-10
    factors = TT_SVD(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)
    
    # TT-ID
    rank_max = max(rank)
    eps = 1e-10
    factors = TT_IDscatter(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-ID is {error}")
    TensorSparseStat(factors)
    
    # TT Cross 
    random_state = 10
    tol = 1e-10
    maxiter = 500
    rank = [1, 30, 30, 1]
    #factors = sparse_ttcross(SpTd, rank, tol, maxiter, random_state)
    factors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors)     
    
    print("Unit test 1 ends!\n")
    return

def testCase2():
    print("Unit test 2 starts!")

    rank = [1, 3, 3, 3, 1]         # TT rank
    order = [5, 5, 5, 5]        # tensor order
    density = [0.5, 0.5, 0.5, 0.5]  # density for every factor
    seed = [2, 3, 4, 5]             # random seeds
    factors = []                    # factor list 
  
    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        np.random.seed(seed[i])
        factor = np.random.random(shape)
        randl = np.random.random(shape)
        factor = np.where(randl>density[i], 0, factor)
        factors.append(factor)                
    SpTd = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
    TensorSparseStat([SpTd])
    
    '''
    outputFlag = 0
    if outputFlag == 1:     
        tnsName = "syn_order_" + "_".join(map(str, order)) + "_synrank_" + "_".join(map(str, rank)) + ".tns"
        tnsPath = "/home/mengzn/Desktop/TensorData/" + tnsName
        cntData = SpT.nnz
        nnzData = SpT.data
        coord = SpT.coords
        with open(tnsPath, "w") as f:
            for i in range(cntData):
                f.write(f"{coord[0][i]} {coord[1][i]} {coord[2][i]} {nnzData[i]}\n")
    '''

    # TT-ID
    rank_max = max(rank)
    eps = 1e-8
    
    #factors = TT_IDscatter(SpTd, rank_max, eps)
    factors_id = TT_IDPRRLDU(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors_id)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-ID is {error}")
    TensorSparseStat(factors_id)

    # TT-SVD
    rank_max = max(rank)
    eps = 1e-8
    factors_svd = TT_SVD(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors_svd)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors_svd)
    
    # TT Cross 
    random_state = 10
    tol = 1e-8
    maxiter = 100
    #factors = sparse_ttcross(SpTd, rank, tol, maxiter, random_state)
    factors_cross = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors_cross)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors_cross)     
    
    print("Unit test 2 ends!\n")
    return

def testCase3():
    print("Unit test 3 starts!")

    rank = [1, 40, 200, 5, 1]          # TT rank
    order = [50, 50, 50, 50]        # tensor order
    density = [1E-2, 1E-2, 1E-2, 1E-1]  # density for every factor
    seed = [100, 200, 300, 400]         # random seeds
    factors = []                        # factor list 
  
    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        np.random.seed(seed[i])
        factor = np.random.random(shape)
        randl = np.random.random(shape)
        factor = np.where(randl>density[i], 0, factor)
        factors.append(factor)                
    SpTd = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
    TensorSparseStat([SpTd])

    # TT-ID
    rank_max = max(rank)
    eps = 1e-6
    factors = TT_IDPRRLDU(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-ID is {error}")
    TensorSparseStat(factors)

    # TT-ID-Forward
    rank_max = max(rank)
    eps = 1e-6
    factors = TT_IDPRRLDU_Forward(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-ID is {error}")
    TensorSparseStat(factors)

    print("Unit test 3 ends!\n")
    return

#testCase1()
#testCase2()
#testCase3()

# Synthetic test: (i) Load -> (ii) Decomp -> (iii) Eval
# (i) Load
FILENAME = "14.tns"
SHAPE = [10, 10, 10, 10]
RANK = [1, 8, 28, 5, 1]
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', '..', '..', 'TensorData', FILENAME)
Tensor = read_from_tns(file_path, SHAPE)
dTensor = Tensor.todense()


# (ii) Decomp and (iii) Eval
# TT-ID
rank_max = 450 #max(RANK)
eps = 1e-8
factors = TT_IDPRRLDU(dTensor, rank_max, eps)
reconT = tl.tt_to_tensor(factors)
error = tl.norm(reconT - dTensor, 2) / tl.norm(dTensor, 2)
print(f"The reconstruction error of TT-ID is {error}")
TensorSparseStat(factors)


# TT-SVD
rank_max = 200
eps = 1e-8
factors = TT_SVD(dTensor, rank_max, eps)
reconT = tl.tt_to_tensor(factors)
error = tl.norm(reconT - dTensor, 2) / tl.norm(dTensor, 2)
print(f"The reconstruction error of TT-SVD is {error}")
TensorSparseStat(factors)


# TT Cross 
random_state = 1
tol = 1e-8
maxiter = 100
RANK = [1, 6, 25, 5, 1]
factors = tensor_train_cross(dTensor, RANK, tol, maxiter, random_state)
reconT = tl.tt_to_tensor(factors)
error = tl.norm(reconT - dTensor, 2) / tl.norm(dTensor, 2)
print(f"The reconstruction error of TT-Cross is {error}")
TensorSparseStat(factors)     

pass
