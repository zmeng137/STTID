import numpy as np
import sparse as sp
import time as tm
import matplotlib.pyplot as plt

# Visualization of 2D matrix 
def view_mat2d(matrix):
    plt.figure(figsize=(10,8))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.tight_layout()
    plt.show()

# Show sparsity information of the input full (dense-format) matrix
def MatrixSparseStat(matrix: np.array):
    print("The sparsity statistics of the input matrix is as follows ...")
    size = matrix.size
    shape = matrix.shape
    cntzero = np.count_nonzero(np.abs(matrix) < 1e-10)
    cntnzero = size - cntzero
    sparsity = cntzero / size
    density = cntnzero / size
    print(f"shape = {shape}, size = {size}, # zero = {cntzero}, sparsity = {sparsity}, # non-zero = {cntnzero}, density = {density}")
    return

# Show sparsity information of the input full (dense-format) tensor array 
def TensorSparseStat(factors: list[np.array], hard_thres = 1e-14):
    # Total number of nonzeros 
    totalnnz = 0
    totalnnz_hthres = 0
    
    # Number of nonzeros of TT-cores
    totalcore_nnz = 0
    totalcore_nnz_hthres = 0
    totalcore_size = 0

    # Count nonzeros of TT-cores 
    for i in range(len(factors)):
        factor = factors[i]
        size = factor.size
        shape = factor.shape

        cntzero = np.count_nonzero(factor == 0)
        cntzero_hthres = np.count_nonzero(np.abs(factor) < hard_thres)
        cntnzero = size - cntzero
        cntnzero_hthres = size - cntzero_hthres

        if len(shape) == 3:
            totalcore_size += size
            totalcore_nnz += cntnzero
            totalcore_nnz_hthres += cntnzero_hthres
        
        totalnnz += cntnzero
        totalnnz_hthres += cntnzero_hthres
        sparsity = cntzero / size
        sparsity_hthres = cntzero_hthres / size
        density = cntnzero / size
        density_hthres = cntnzero_hthres / size 
        
        print(f"Tensor {i}: shape = {shape}, size = {size}, # zero = {cntzero}, sparsity = {sparsity}, # non-zero = {cntnzero}, density = {density}..")
        print(f"..If applying hard threshold {hard_thres}, # zero = {cntzero_hthres}, sparsity = {sparsity_hthres}, # non-zero = {cntnzero_hthres}, density = {density_hthres}")
    
    print(f"\nTotal number of non-zero (hard threshold {hard_thres}): {totalnnz} ({totalnnz_hthres}).") 
    print(f"Average density of the output tensor train: {totalcore_nnz/totalcore_size} ({totalcore_nnz_hthres/totalcore_size})") 

    return 

# Find Pct%-close-to-0 values of a martrix and cast them to 0
def CastValueAroundZero(Mat: np.array, Pct: float) -> np.array:
    absMat = np.abs(Mat)
    eleCnt = Mat.size
    # Sort and cast
    sortIndx = np.unravel_index(np.argsort(absMat, axis=None), absMat.shape)
    sortMat = absMat[sortIndx]
    if Pct >= 1.0 or Pct <= 0.0:
        print("The input percentage should be between 0 and 1. The percentage is set to 0.5 by default.")
        Pct = 0.5
    thresIdx = int(eleCnt * Pct) 
    thresVal = sortMat[thresIdx]    
    Mat = np.where(absMat > thresVal, Mat, 0)
    return Mat

# Read FROSTT data
def readfrostt(path: str, shape: tuple):
    print(f"Start loading tensor data {path}...")
    start_t = tm.time()
    #is_it_equal_function = lambda x:x in shape
    shape = np.array(shape)
    with open(path, "r") as file:
        modeList = []
        for line in file:
            numbers = line.strip().split()
            numbers = np.array([int(num) for num in numbers])
            coords = numbers[:-1]-1
            cmp = coords < shape
            if np.all(cmp):
                modeList.append(numbers)
        modeList = np.array(modeList, dtype=np.int32)
        modeList = modeList.T
        data = modeList[-1] 
        coords = modeList[:-1]-1
        spt = sp.COO(coords, data, shape)
    end_t = tm.time()
    print(f"Finish loading! It took {end_t - start_t} seconds.")
    return spt

def read_from_tns(filename, shape):
    data_array = np.loadtxt(filename) # Load all data using numpy (fast method)
    
    # Split into coordinates and values
    coords = data_array[:, :-1].astype(int)  # All columns except last as coordinates
    values = data_array[:, -1]  # Last column as values
    
    # Create COO array
    return sp.COO(coords.T, values, shape=shape)