#include "structures.h"
#include "util.h"
#include "cutil.h"

// CUDA Thrust
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>

#include <mkl/mkl_spblas.h>

// Convert un-row-sorted COO format to row-sorted COO format by cuSparse. All changes happen in place. 
void cusparse_dcooRowSort(int num_rows, int num_columns, int nnz, int* d_rows, int* d_columns, double* d_values)
{
    // Device memory management
    int* d_permutation;
    double* d_values_sorted;
    void* d_buffer;
    size_t bufferSize;
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_values_sorted, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_permutation,   nnz * sizeof(size_t)));

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    // Create sparse vector for the permutation
    util::CHECK_CUSPARSE_ERROR(cusparseCreateSpVec(&vec_permutation, nnz, nnz,
        d_permutation, d_values_sorted, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    // Create dense vector for wrapping the original coo values
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vec_values, nnz, d_values, CUDA_R_64F));

    // Query working space of COO sort
    util::CHECK_CUSPARSE_ERROR(cusparseXcoosort_bufferSizeExt(handle, num_rows,
        num_columns, nnz, d_rows, d_columns, &bufferSize));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_buffer, bufferSize));

    // Setup permutation vector to identity
    util::CHECK_CUSPARSE_ERROR(cusparseCreateIdentityPermutation(handle, nnz, d_permutation));
    util::CHECK_CUSPARSE_ERROR(cusparseXcoosortByRow(handle, num_rows, num_columns, nnz, d_rows, d_columns, d_permutation, d_buffer));
    util::CHECK_CUSPARSE_ERROR(cusparseGather(handle, vec_values, vec_permutation));

    // (?) Copy sorted value to value array
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_values ,d_values_sorted ,nnz * sizeof(double) , cudaMemcpyDeviceToDevice));

    // destroy matrix/vector descriptors
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpVec(vec_permutation));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vec_values));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));

    // Device memory release
    util::CHECK_CUDART_ERROR(cudaFree(d_permutation));
    util::CHECK_CUDART_ERROR(cudaFree(d_values_sorted));
    util::CHECK_CUDART_ERROR(cudaFree(d_buffer));
    return;
}

// Convert un-col-sorted COO format to col-sorted COO format by cuSparse. All changes happen in place. 
void cusparse_dcooColSort(int num_rows, int num_columns, int nnz, int* d_rows, int* d_columns, double* d_values)
{
    // Device memory management
    int* d_permutation;
    double* d_values_sorted;
    void* d_buffer;
    size_t bufferSize;
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_values_sorted, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_permutation,   nnz * sizeof(size_t)));

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    // Create sparse vector for the permutation
    util::CHECK_CUSPARSE_ERROR(cusparseCreateSpVec(&vec_permutation, nnz, nnz,
        d_permutation, d_values_sorted, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    // Create dense vector for wrapping the original coo values
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vec_values, nnz, d_values, CUDA_R_64F));

    // Query working space of COO sort
    util::CHECK_CUSPARSE_ERROR(cusparseXcoosort_bufferSizeExt(handle, num_rows,
        num_columns, nnz, d_rows, d_columns, &bufferSize));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_buffer, bufferSize));

    // Setup permutation vector to identity
    util::CHECK_CUSPARSE_ERROR(cusparseCreateIdentityPermutation(handle, nnz, d_permutation));
    util::CHECK_CUSPARSE_ERROR(cusparseXcoosortByColumn(handle, num_rows, num_columns, nnz, d_rows, d_columns, d_permutation, d_buffer));
    util::CHECK_CUSPARSE_ERROR(cusparseGather(handle, vec_values, vec_permutation));

    // (?) Copy sorted value to value array
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_values ,d_values_sorted ,nnz * sizeof(double) , cudaMemcpyDeviceToDevice));
    
    // destroy matrix/vector descriptors
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpVec(vec_permutation));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vec_values));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));

    // Device memory release
    util::CHECK_CUDART_ERROR(cudaFree(d_permutation));
    util::CHECK_CUDART_ERROR(cudaFree(d_values_sorted));
    util::CHECK_CUDART_ERROR(cudaFree(d_buffer));
    return;
}

// Convert un-row-sorted COO format to CSR format by cuSparse.
void cusparse_dcoo2csr(int num_rows, int num_columns, int nnz, int* d_cooRows, int* d_cooCols, double* d_cooVals, int* d_csrRows)
{
    // Sort row
    cusparse_dcooRowSort(num_rows, num_columns, nnz, d_cooRows, d_cooCols, d_cooVals);

    // Convert COO to CSR
    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));
    util::CHECK_CUSPARSE_ERROR(cusparseXcoo2csr(handle, d_cooRows, nnz, num_rows, d_csrRows, CUSPARSE_INDEX_BASE_ZERO));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));

    return;
}

// Convert sorted COO format to CSC format by cuSparse.
void cusparse_dcoo2csc(int num_rows, int num_columns, int nnz, 
    int* d_cooRows, int* d_cooCols, double* d_cooVals, int* d_cscCols)
{
    // Sort column
    cusparse_dcooColSort(num_rows, num_columns, nnz, d_cooRows, d_cooCols, d_cooVals);

    // Convert COO to CSR
    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));
    util::CHECK_CUSPARSE_ERROR(cusparseXcoo2csr(handle, d_cooCols, nnz, num_columns, d_cscCols, CUSPARSE_INDEX_BASE_ZERO));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));

    return;
}

// 64-bit upper triangular solver cusparse
void cusparse_uptrsv_csr_64(int64_t rows, int64_t cols, int64_t nnz, int64_t* d_csrRowPtr, int64_t* d_csrColInd, 
    double* d_csrVal, double* d_b, double* d_x, double alpha)
{
    assertm(rows == cols, "must be square matrix!");

    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&matA, rows, cols, nnz,
    d_csrRowPtr, d_csrColInd, d_csrVal,
    CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_UPPER;
    util::CHECK_CUSPARSE_ERROR(cusparseSpMatSetAttribute(matA, 
            CUSPARSE_SPMAT_FILL_MODE,
            &fillMode,
            sizeof(cusparseFillMode_t)));

    cusparseDnVecDescr_t vecX, vecB;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_64F));
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecB, cols, d_b, CUDA_R_64F));

    cusparseSpSVDescr_t spsvDescr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_createDescr(&spsvDescr));

    size_t bufferSize;
    void* buffer = nullptr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        &bufferSize));
    cudaMalloc(&buffer, bufferSize);

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        buffer));

    {util::Timer timer("cuSpSV solve");
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecB, vecX, CUDA_R_64F,
    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));}

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_destroyDescr(spsvDescr));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecB));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));
    util::CHECK_CUDART_ERROR(cudaFree(buffer));
}

void cusparse_uptrsv_csr_32(int rows, int cols, int nnz, int* d_csrRowPtr, int* d_csrColInd, 
    double* d_csrVal, double* d_b, double* d_x, double alpha)
{
    assertm(rows == cols, "must be square matrix!");

    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&matA, rows, cols, nnz,
    d_csrRowPtr, d_csrColInd, d_csrVal,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_UPPER;
    util::CHECK_CUSPARSE_ERROR(cusparseSpMatSetAttribute(matA, 
            CUSPARSE_SPMAT_FILL_MODE,
            &fillMode,
            sizeof(cusparseFillMode_t)));

    cusparseDnVecDescr_t vecX, vecB;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_64F));
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecB, cols, d_b, CUDA_R_64F));

    cusparseSpSVDescr_t spsvDescr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_createDescr(&spsvDescr));

    size_t bufferSize;
    void* buffer = nullptr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        &bufferSize));
    cudaMalloc(&buffer, bufferSize);

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        buffer));

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecB, vecX, CUDA_R_64F,
    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_destroyDescr(spsvDescr));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecB));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));
    util::CHECK_CUDART_ERROR(cudaFree(buffer));
}

void cud_coo2cs(const int64_t num_rows, const int64_t num_cols, int64_t nnz, 
    int64_t* d_cooRows, int64_t* d_cooCols, double* d_cooVals, int64_t* d_csPtrs, bool row_order = CSRT)
{
    if (row_order) {
        // Sort by row
        thrust::stable_sort_by_key(thrust::device, d_cooCols, d_cooCols + nnz, thrust::make_zip_iterator(thrust::make_tuple(d_cooRows, d_cooVals)));
        thrust::stable_sort_by_key(thrust::device, d_cooRows, d_cooRows + nnz, thrust::make_zip_iterator(thrust::make_tuple(d_cooCols, d_cooVals)));
        
        // Create temporary storage for counting
        thrust::device_vector<int64_t> d_row_counts(num_rows, 0);
        thrust::device_vector<int64_t> d_unique_rows(num_rows);

        // Count entries per row using reduce_by_key
        auto ones_begin = thrust::make_constant_iterator<int64_t>(1);
        
        // d_cooRows are the keys, we count ones for each key
        auto end = thrust::reduce_by_key(thrust::device, d_cooRows, d_cooRows + nnz, ones_begin, d_unique_rows.begin(), d_row_counts.begin());

        // Initialize csrPtrs to zero
        thrust::fill(thrust::device, d_csPtrs, d_csPtrs + num_rows + 1, 0);

        // Copy counts to appropriate positions in csrPtrs
        thrust::scatter(thrust::device, d_row_counts.begin(), d_row_counts.begin() + (end.first - d_unique_rows.begin()), d_unique_rows.begin(), d_csPtrs);

        // Compute exclusive prefix sum to get final row pointers
        thrust::exclusive_scan(thrust::device, d_csPtrs, d_csPtrs + num_rows + 1, d_csPtrs);
    } else {
        // Sort by col
        thrust::stable_sort_by_key(thrust::device, d_cooRows, d_cooRows + nnz, thrust::make_zip_iterator(thrust::make_tuple(d_cooCols, d_cooVals)));
        thrust::stable_sort_by_key(thrust::device, d_cooCols, d_cooCols + nnz, thrust::make_zip_iterator(thrust::make_tuple(d_cooRows, d_cooVals)));
        
        // Create temporary storage for counting
        thrust::device_vector<int64_t> d_col_counts(num_cols, 0);
        thrust::device_vector<int64_t> d_unique_cols(num_cols);

        // Count entries per row using reduce_by_key
        auto ones_begin = thrust::make_constant_iterator<int64_t>(1);
        
        // d_cooRows are the keys, we count ones for each key
        auto end = thrust::reduce_by_key(thrust::device, d_cooCols, d_cooCols + nnz, ones_begin, d_unique_cols.begin(), d_col_counts.begin());

        // Initialize csrPtrs to zero
        thrust::fill(thrust::device, d_csPtrs, d_csPtrs + num_cols + 1, 0);

        // Copy counts to appropriate positions in csrPtrs
        thrust::scatter(thrust::device, d_col_counts.begin(), d_col_counts.begin() + (end.first - d_unique_cols.begin()), d_unique_cols.begin(), d_csPtrs);

        // Compute exclusive prefix sum to get final row pointers
        thrust::exclusive_scan(thrust::device, d_csPtrs, d_csPtrs + num_rows + 1, d_csPtrs);
    }

    return;
}

// Assign values to dense b vector (device kernel)
__global__ void b_assign_kernel(double* d_b, int64_t pos, int64_t* d_B_cscCols, int64_t* d_B_cooRows, double* d_B_cooVals)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row_end = d_B_cscCols[pos + 1];
    int64_t row_start = d_B_cscCols[pos];
    int64_t n = row_end - row_start;
    while (tid < n) {
        d_b[d_B_cooRows[row_start + tid]] = d_B_cooVals[row_start + tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void int64cast_kernel(int* d_array_32, int64_t* d_array_64, int N) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < N) {
        d_array_32[tid] = d_array_64[tid];
        tid += blockDim.x * gridDim.x;
    }
}

// Assign values to dense b vector (call from host)
void denseb_assign_gpu(double* d_b, int64_t pos, int64_t* d_B_cscCols, int64_t* d_B_cooRows, double* d_B_cooVals)
{
    b_assign_kernel<<<128, 128>>>(d_b, pos, d_B_cscCols, d_B_cooRows, d_B_cooVals);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();
}

__global__ void convertLongLongToInt64(const long long* input, int64_t* output, size_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < size) {
        output[idx] = (int64_t)input[idx];
        idx += blockDim.x * gridDim.x;
    }
}

void cusparse_trsv_idkernel(
    decompRes::SparseInterpRes<double> idResult, 
    decompRes::SparsePrrlduRes<double> prrlduResult,
    long long output_rank, long long Nc)
{
    // Sparse-COO U11 -> CSR U11
    int64_t U11_rows = prrlduResult.sparse_U11.rows;
    int64_t U11_cols = prrlduResult.sparse_U11.cols;
    int64_t U11_nnz = prrlduResult.sparse_U11.nnz_count;
    int64_t* d_U11_cooRows, * d_U11_cooCols, * d_U11_csrPtr;
    double* d_U11_cooVals;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_U11_csrPtr, (U11_rows + 1) * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_U11_cooRows, U11_nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_U11_cooCols, U11_nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_U11_cooVals, U11_nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_U11_cooRows, prrlduResult.sparse_U11.row_indices, U11_nnz * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_U11_cooCols, prrlduResult.sparse_U11.col_indices, U11_nnz * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_U11_cooVals, prrlduResult.sparse_U11.values, U11_nnz * sizeof(double), cudaMemcpyHostToDevice));
    cud_coo2cs(U11_rows, U11_cols, U11_nnz, d_U11_cooRows, d_U11_cooCols, d_U11_cooVals, d_U11_csrPtr, CSRT);

    // Sparse-COO B -> CSC B
    int64_t B_rows = prrlduResult.sparse_B.rows;
    int64_t B_cols = prrlduResult.sparse_B.cols;
    int64_t B_nnz = prrlduResult.sparse_B.nnz_count;
    int64_t* d_B_cooRows, * d_B_cooCols, * d_B_cscPtr;
    double* d_B_cooVals;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_B_cscPtr, (B_cols + 1) * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_B_cooRows, B_nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_B_cooCols, B_nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_B_cooVals, B_nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_B_cooRows, prrlduResult.sparse_B.row_indices, B_nnz * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_B_cooCols, prrlduResult.sparse_B.col_indices, B_nnz * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_B_cooVals, prrlduResult.sparse_B.values, B_nnz * sizeof(double), cudaMemcpyHostToDevice));
    cud_coo2cs(B_rows, B_cols, B_nnz, d_B_cooRows, d_B_cooCols, d_B_cooVals, d_B_cscPtr, CSCT);

    // Dense vector memory allocation
    double* x = new double[output_rank]{0.0};
    double* d_b, *d_x;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_b, output_rank * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_x, output_rank * sizeof(double)));

    // Compute the interpolative coefficients through solving upper triangular systems
    std::cout << "Sptrsv (CuSparse trsv) for coefficients starts.\n";
    for (int64_t i = 0; i < Nc - output_rank; ++i) {
        // Right-hand side b (one column of the U)
        util::CHECK_CUDART_ERROR(cudaMemset(d_b , 0, sizeof(double) * output_rank));
        denseb_assign_gpu(d_b, i, d_B_cscPtr, d_B_cooRows, d_B_cooVals);

        // CuSparse 64-bit triangular solver
        cusparse_uptrsv_csr_64(U11_rows, U11_cols, U11_nnz, d_U11_csrPtr, d_U11_cooCols, d_U11_cooVals, d_b, d_x, 1.0);
    
        // Copy x
        util::CHECK_CUDART_ERROR(cudaMemcpy(x, d_x, output_rank * sizeof(double), cudaMemcpyDeviceToHost));

        // Copy the solution to iU11 columns
        for (int64_t j = 0; j < output_rank; ++j) 
            idResult.interp_coeff[j * (Nc - output_rank) + i] = x[j];       
    }

    delete[] x;
    util::CHECK_CUDART_ERROR(cudaFree(d_x));
    util::CHECK_CUDART_ERROR(cudaFree(d_b));
    util::CHECK_CUDART_ERROR(cudaFree(d_U11_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_U11_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_U11_csrPtr));
    util::CHECK_CUDART_ERROR(cudaFree(d_U11_cooVals));
    util::CHECK_CUDART_ERROR(cudaFree(d_B_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_B_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_B_cscPtr));
    util::CHECK_CUDART_ERROR(cudaFree(d_B_cooVals));
    std::cout << "Sptrsv (CuSparse trsv) for coefficients ends.\n"; 
}

void cusparse_trsv_idkernel_2(
    decompRes::SparseInterpRes<double> idResult, 
    decompRes::SparsePrrlduRes<double> prrlduResult,
    long long output_rank, long long Nc) 
{
    // Sparse-COO U11 -> CSR U11
    long long U11_rows = prrlduResult.sparse_U11.rows;
    long long U11_cols = prrlduResult.sparse_U11.cols;
    long long U11_nnz = prrlduResult.sparse_U11.nnz_count;
    long long* U11_cooRows = prrlduResult.sparse_U11.row_indices;
    long long* U11_cooCols = prrlduResult.sparse_U11.col_indices;
    double* U11_cooVals = prrlduResult.sparse_U11.values;
    sparse_matrix_t cooU11, csrU11;
    util::CHECK_MKL_ERROR(mkl_sparse_d_create_coo(&cooU11, SPARSE_INDEX_BASE_ZERO, 
        U11_rows, U11_cols, U11_nnz, U11_cooRows, U11_cooCols, U11_cooVals));    
    util::CHECK_MKL_ERROR(mkl_sparse_convert_csr(cooU11, SPARSE_OPERATION_NON_TRANSPOSE, &csrU11));
    sparse_index_base_t indexing_u;
    long long *csrU11_row_start;
    long long *csrU11_row_end;
    long long *csrU11_col_ind;
    double *csrU11_values;
    util::CHECK_MKL_ERROR(mkl_sparse_d_export_csr(csrU11, &indexing_u, &U11_rows, &U11_cols,  
        &csrU11_row_start, &csrU11_row_end, &csrU11_col_ind, &csrU11_values));

    // Sparse-COO B -> CSC B
    long long B_rows = prrlduResult.sparse_B.rows;
    long long B_cols = prrlduResult.sparse_B.cols;
    long long B_nnz = prrlduResult.sparse_B.nnz_count;
    long long* B_cooRows = prrlduResult.sparse_B.row_indices;
    long long* B_cooCols = prrlduResult.sparse_B.col_indices;
    double* B_cooVals = prrlduResult.sparse_B.values;
    sparse_matrix_t cooB, cscB;
    util::CHECK_MKL_ERROR(mkl_sparse_d_create_coo(&cooB, SPARSE_INDEX_BASE_ZERO, 
        B_rows, B_cols, B_nnz, B_cooRows, B_cooCols, B_cooVals));    
    util::CHECK_MKL_ERROR(mkl_sparse_convert_csr(cooB, SPARSE_OPERATION_TRANSPOSE, &cscB));
    sparse_index_base_t indexing_b;
    long long *cscB_col_start;
    long long *cscB_col_end;
    long long *cscB_row_ind;
    double *cscB_values;
    util::CHECK_MKL_ERROR(mkl_sparse_d_export_csr(cscB, &indexing_b, &B_rows, &B_cols,  
        &cscB_col_start, &cscB_col_end, &cscB_row_ind, &cscB_values));
    
    double* b = new double[output_rank]{0.0};
    double* x = new double[output_rank]{0.0};
    
    // Device memory allocation
    int64_t* d_csrU11_row_start;
    int64_t* d_csrU11_col_ind;
    double* d_csrU11_values, *d_b, *d_x;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_row_start, (U11_rows + 1) * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_col_ind, U11_nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_values, U11_nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_b, output_rank * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_x, output_rank * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrU11_row_start, csrU11_row_start, (U11_rows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrU11_col_ind, csrU11_col_ind, U11_nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrU11_values, csrU11_values, U11_nnz * sizeof(double), cudaMemcpyHostToDevice));


    int* d_csrU11_row_start_32;
    int* d_csrU11_col_ind_32;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_row_start_32, (U11_rows + 1) * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_col_ind_32, U11_nnz * sizeof(int)));
    int64cast_kernel<<<128,128>>>(d_csrU11_row_start_32, d_csrU11_row_start, U11_rows + 1);
    cudaDeviceSynchronize();
    int64cast_kernel<<<128,128>>>(d_csrU11_col_ind_32, d_csrU11_col_ind, U11_nnz);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();

    // Compute the interpolative coefficients through solving upper triangular systems
    std::cout << "Sptrsv (CuSparse trsv) for coefficients starts.\n";
    for (long long i = 0; i < Nc - output_rank; ++i) {
        // Right hand side b (one column of the U)
        memset(b, 0, sizeof(double) * output_rank);
        long long row_end = cscB_col_start[i + 1];
        long long row_start = cscB_col_start[i];
        for (long long j = 0; j < row_end - row_start; ++j) {
            b[cscB_row_ind[row_start + j]] = cscB_values[row_start + j];
        }
        
        // Copy b
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_b, b, output_rank * sizeof(double), cudaMemcpyHostToDevice));

        // CuSparse 64-bit triangular solver
        cusparse_uptrsv_csr_64(U11_rows, U11_cols, U11_nnz, d_csrU11_row_start, d_csrU11_col_ind, d_csrU11_values, d_b, d_x, 1.0);
        //cusparse_uptrsv_csr_32(U11_rows, U11_cols, U11_nnz, d_csrU11_row_start_32, d_csrU11_col_ind_32, d_csrU11_values, d_b, d_x, 1.0);}
        
        // Copy x
        util::CHECK_CUDART_ERROR(cudaMemcpy(x, d_x, output_rank * sizeof(double), cudaMemcpyDeviceToHost));

        // Copy the solution to iU11 columns
        for (long long j = 0; j < output_rank; ++j) 
            idResult.interp_coeff[j * (Nc - output_rank) + i] = x[j];       
    }
    
    // Clean up
    cudaFree(d_csrU11_row_start_32);
    cudaFree(d_csrU11_col_ind_32);

    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cooU11));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(csrU11));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cooB));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cscB));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrU11_row_start));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrU11_col_ind));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrU11_values));
    util::CHECK_CUDART_ERROR(cudaFree(d_b));
    util::CHECK_CUDART_ERROR(cudaFree(d_x));
    delete[] b;
    delete[] x;
    std::cout << "Sptrsv (CuSparse trsv) for coefficients ends.\n";      
}

void cusparse_trsv_idkernel_3(
    decompRes::SparseInterpRes<double> idResult, 
    decompRes::SparsePrrlduRes<double> prrlduResult,
    long long output_rank, long long Nc) 
{
    // Sparse-COO U11 -> CSR U11
    long long U11_rows = prrlduResult.sparse_U11.rows;
    long long U11_cols = prrlduResult.sparse_U11.cols;
    long long U11_nnz = prrlduResult.sparse_U11.nnz_count;
    long long* U11_cooRows = prrlduResult.sparse_U11.row_indices;
    long long* U11_cooCols = prrlduResult.sparse_U11.col_indices;
    double* U11_cooVals = prrlduResult.sparse_U11.values;
    sparse_matrix_t cooU11, csrU11;
    util::CHECK_MKL_ERROR(mkl_sparse_d_create_coo(&cooU11, SPARSE_INDEX_BASE_ZERO, 
        U11_rows, U11_cols, U11_nnz, U11_cooRows, U11_cooCols, U11_cooVals));    
    util::CHECK_MKL_ERROR(mkl_sparse_convert_csr(cooU11, SPARSE_OPERATION_NON_TRANSPOSE, &csrU11));
    sparse_index_base_t indexing_u;
    long long *csrU11_row_start;
    long long *csrU11_row_end;
    long long *csrU11_col_ind;
    double *csrU11_values;
    util::CHECK_MKL_ERROR(mkl_sparse_d_export_csr(csrU11, &indexing_u, &U11_rows, &U11_cols,  
        &csrU11_row_start, &csrU11_row_end, &csrU11_col_ind, &csrU11_values));

    // Sparse-COO B -> CSC B
    long long B_rows = prrlduResult.sparse_B.rows;
    long long B_cols = prrlduResult.sparse_B.cols;
    long long B_nnz = prrlduResult.sparse_B.nnz_count;
    long long* B_cooRows = prrlduResult.sparse_B.row_indices;
    long long* B_cooCols = prrlduResult.sparse_B.col_indices;
    double* B_cooVals = prrlduResult.sparse_B.values;
    sparse_matrix_t cooB, cscB;
    util::CHECK_MKL_ERROR(mkl_sparse_d_create_coo(&cooB, SPARSE_INDEX_BASE_ZERO, 
        B_rows, B_cols, B_nnz, B_cooRows, B_cooCols, B_cooVals));    
    util::CHECK_MKL_ERROR(mkl_sparse_convert_csr(cooB, SPARSE_OPERATION_TRANSPOSE, &cscB));
    sparse_index_base_t indexing_b;
    long long *cscB_col_start;
    long long *cscB_col_end;
    long long *cscB_row_ind;
    double *cscB_values;
    util::CHECK_MKL_ERROR(mkl_sparse_d_export_csr(cscB, &indexing_b, &B_rows, &B_cols,  
        &cscB_col_start, &cscB_col_end, &cscB_row_ind, &cscB_values));
    
    double* b = new double[output_rank]{0.0};
    double* x = new double[output_rank]{0.0};
    
    // Device memory allocation
    int64_t* d_csrU11_row_start;
    int64_t* d_csrU11_col_ind;
    double* d_csrU11_values, *d_b, *d_x;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_row_start, (U11_rows + 1) * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_col_ind, U11_nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrU11_values, U11_nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_b, output_rank * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_x, output_rank * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrU11_row_start, csrU11_row_start, (U11_rows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrU11_col_ind, csrU11_col_ind, U11_nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrU11_values, csrU11_values, U11_nnz * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&matA, U11_rows, U11_cols, U11_nnz,
        d_csrU11_row_start, d_csrU11_col_ind, d_csrU11_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_UPPER;
    util::CHECK_CUSPARSE_ERROR(cusparseSpMatSetAttribute(matA, 
            CUSPARSE_SPMAT_FILL_MODE,
            &fillMode,
            sizeof(cusparseFillMode_t)));

    cusparseDnVecDescr_t vecX, vecB;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecX, U11_cols, d_x, CUDA_R_64F));
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecB, U11_cols, d_b, CUDA_R_64F));

    cusparseSpSVDescr_t spsvDescr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_createDescr(&spsvDescr));

    double alpha = 1.0;
    size_t bufferSize;
    void* buffer = nullptr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        &bufferSize));
    cudaMalloc(&buffer, bufferSize);

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        buffer));
    
    // Compute the interpolative coefficients through solving upper triangular systems
    std::cout << "Sptrsv (CuSparse trsv) for coefficients starts.\n";
    for (long long i = 0; i < Nc - output_rank; ++i) {
        // Right hand side b (one column of the U)
        memset(b, 0, sizeof(double) * output_rank);
        long long row_end = cscB_col_start[i + 1];
        long long row_start = cscB_col_start[i];
        long long interval = row_end - row_start;
        for (long long j = 0; j < interval; ++j) {
            b[cscB_row_ind[row_start + j]] = cscB_values[row_start + j];
        }
        
        // Copy b
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_b, b, output_rank * sizeof(double), cudaMemcpyHostToDevice));

        // CuSparse 64-bit triangular solver
        {util::Timer timer("cuSpSV solves");
        util::CHECK_CUSPARSE_ERROR(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecB, vecX, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));}
        
        // Copy x
        util::CHECK_CUDART_ERROR(cudaMemcpy(x, d_x, output_rank * sizeof(double), cudaMemcpyDeviceToHost));

        // Copy the solution to iU11 columns
        for (long long j = 0; j < output_rank; ++j) 
            idResult.interp_coeff[j * (Nc - output_rank) + i] = x[j];       
    }
    
    // Clean up
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_destroyDescr(spsvDescr));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecB));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));
    util::CHECK_CUDART_ERROR(cudaFree(buffer));

    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cooU11));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(csrU11));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cooB));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cscB));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrU11_row_start));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrU11_col_ind));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrU11_values));
    util::CHECK_CUDART_ERROR(cudaFree(d_b));
    util::CHECK_CUDART_ERROR(cudaFree(d_x));
    delete[] b;
    delete[] x;
    std::cout << "Sptrsv (CuSparse trsv) for coefficients ends.\n";      
}