# STTID: High-Performance Sparse Tensor-Train Interpolative Decomposition

## Overview

STTID is a high-performance sparse implementation of the Tensor-Train Interpolative Decomposition (TT-ID) algorithm. By leveraging interpolative decomposition (ID) based on partial rank-revealing LU decomposition (PRRLU), TT-ID enables sparse computation of the tensor-train (TT) decomposition for sparse tensors while preserving sparsity in the resulting TT-cores. STTID is designed to exploit the sparse format of input tensors, incorporating multiple-level optimization techniques and GPU parallelization to enhance computational performance.

## Implementations

Two implementations are included:

**Python** — Provides a numerical TT-ID algorithm in dense format, along with other tensor-train algorithms such as TT-SVD (with/without Lasso) and TT-cross (QR-based and LU-based). Suitable for rapid quality comparisons including core density, TT-ranks, and reconstruction error between TT-ID and other TT methods.

**C++/CUDA** — Offers the full STTID implementation with enhanced performance for processing much larger sparse tensors. Recommended for large-scale tensor-train processing.

## Dependencies

### Python Requirements

- NumPy
- SciPy
- Sparse
- [TensorLy](https://tensorly.org/stable/index.html) v0.9.0

### C++/CUDA Requirements

- C++17 standard — g++ 9.4.0 and NVCC 12.0 or above
- CMake 3.23.1 or above
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html) 2020.0.0 (Sparse BLAS)
- [CUDA 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive) (including cuSPARSE 12.1 and cuBLAS 12.0.2)
- TBLIS 1.3.0 (if one wants to build the dense TT-ID implementation as well.)

## Data

In `Data/`, we provide the scripts used for generating random sparse tensors and knowledge graph tensors. The generated tensors are already provided in `Data/random_tensor/` and `Data/kgraph_tensor/`. For FROSTT tensors, please refer to https://frostt.io/.

## Running TT-ID in Python

The `Py/` directory contains implementations of TT-ID, TT-SVD, and TT-cross. To compare the quality of different TT methods, run:

```shell
pip install -r requirements.txt
python test_quality.py
```

## Building and Running STTID (C++/CUDA)

Navigate to `STTID/` and run `build.sh` to build codes with CMake. `build.sh` builds STTID and dense TT-ID respectively. One can optionally choose what to build by commenting the related commands. 

This produces several executables in the `STTID/exe/` directory:

| Executable | Description |
|---|---|
| `sttid_dim4_cpu, sttid_dim4_gpu` | STTID for order-4 tensors (CPU and GPU implementation) |
| `sttid_dim5_cpu, sttid_dim5_gpu` | STTID for order-5 tensors (CPU and GPU implementation) |
| `ttid_dim4_cpu, ttid_dim5_cpu` | Dense TT-ID for order-4 and order-5 tensors (CPU) |

### Usage 

```shell
./sttid_dim4_cpu [data_file_path] [nnz] [order1] [order2] [order3] [order4] [rmax] [epsilon] [spthres] [binary] [idx_offset] 
./sttid_dim5_cpu [data_file_path] [nnz] [order1] [order2] [order3] [order4] [order5] [rmax] [epsilon] [spthres] [binary] [idx_offset]
```

**Parameters:**

| Parameter | Description |
|---|---|
| `data_file_path` | Path to the input tensor data file |
| `nnz` | Number of nonzeros in the sparse tensor |
| `order1 ... orderN` | Tensor shape (one dimension per mode) |
| `rmax` | Maximum TT-ranks |
| `epsilon` | Approximation tolerance |
| `spthres` | Sparse threshold value |
| `binary` | For knowledge graph tensors whose entries are 1 or 0 and not stored explicitly in the .tns|
| `idx_offset` | For some tensors whose indices do not start from 0 |

We provide templates to run different executables on various tensors in `STTID/run.sh`. 