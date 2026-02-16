# STTID: High-Performance Sparse Tensor-Train Interpolative Decomposition
## Overview
STTID is a high-performance sparse implementation of a Tensor-Train Interpolative Decomposition (TT-ID) algorithm. By using interpolative decomposition (ID) based on partial rank-revealing LU decomposition (PRRLU), TT-ID enables sparse computation of the tensor-train (TT) decomposition for sparse tensors while preserving sparsity in the resulting TT-cores. STTID is designed to leverage the sparse format of input tensors in TT-ID, incorporating various optimization techniques and GPU parallelization to enhance computational performance.

## Installation
Two implementations are included: a Python version and a C++/CUDA version. The Python implementation provides a numerical TT-ID algorithm in dense format, along with other tensor-train (TT) algorithms such as TT-SVD and TT-cross. The C++/CUDA implementation offers the STTID with enhanced performance to process much larger sparse tensors. For quality comparisons - including core density, TT-ranks, and reconstruction error between TT-ID and other TT methods - the Python implementation is suitable for rapid testing. For large-scale tensor train processing, one can use the C++/CUDA STTID implementation. 

### Python Requirements
In addition to fundamental libraries like NumPy and SciPy, we utilize TensorLy (v0.9.0) for convenient tensor operations (https://tensorly.org/stable/index.html).

### C++ Requirements
- C++ 17 standard, g++ 9.4.0 and NVCC 12.0 or above.
- CMake 3.23.1 or above.
- Sparse BLAS from Intel MKL 2020.0.0 (https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)
- CUDA 12.0 (https://developer.nvidia.com/cuda-12-0-0-download-archive, including cuSPARSE 12.1 and cuBLAS 12.0.2).

## Run TT-ID in Python
The Py/ directory contains implementations of TT-ID, TT-SVD, and TT-cross (from TensorLy). Users can directly compare the quality of different TT methods by executing test_quality.py. The package also includes a synthetic tensor generation script for testing purposes.

## Build and Run STTID
Under STTID/Cpp/
```shell
mkdir build
cd build
cmake ..
```
After generating the Makefile using CMake, users can compile the code by simply running make. This will produce several executable files in the synthetic_test/ directory.
- ./synthetic_sparse_test_dim4 [data_file_path] [nnz] [order1] [order2] [order3] [order4] [rmax] [epsilon] [spthres]. 
- ./synthetic_sparse_test_dim5 [data_file_path] [nnz] [order1] [order2] [order3] [order4] [order5] [rmax] [epsilon] [spthres].
- ./synthetic_sparse_test_dim4_ls [data_file_path] [nnz] [order1] [order2] [order3] [order4] [rmax] [epsilon] [spthres]. 
- ./synthetic_sparse_test_dim5_ls [data_file_path] [nnz] [order1] [order2] [order3] [order4] [order5] [rmax] [epsilon] [spthres].
  
The executables handle STTID processing for order-4 and order-5 tensors respectively. Executables suffixed with '_ls' process tensors with extremely large shapes using a hash table-based sparse vector technique. Users must specify the following parameters:
1. Input data path (file location of the tensor data)
2. Number of nonzeros in the sparse tensor
3. Tensor shape (dimensions for each order)
4. Maximum TT-ranks
5. Approximation tolerance
6. Sparse threshold value
   
..
mit license..
