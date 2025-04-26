#include <gtest/gtest.h>
#include "gpu_kernel.h"
#include "util.h"
#include "cutil.h"

TEST(ThrustSpOps, coo2csr_1)
{
    // COO format data
    int64_t m = 4;  // matrix size
    int64_t n = 5;
    int64_t nnz = 7;  // number of non-zeros
    
    // COO arrays
    int64_t h_cooRows[] = {1, 2, 0, 2, 2, 0, 1};
    int64_t h_cooCols[] = {1, 2, 1, 3, 1, 0, 4};
    double h_cooVals[] = {3.0, 4.0, 2.0, 5.0, 6.0, 1.0, 7.0};

    // Host CSR pointer array
    int64_t* h_csrRows = new int64_t[m + 1];

    // Device arrays for COO and CSR
    int64_t *d_cooRows, *d_cooCols, *d_csrRows;
    double *d_cooVals;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooRows, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooCols, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooVals, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRows, (m + 1) * sizeof(int64_t)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooVals, h_cooVals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    cud_coo2cs(m, n, nnz, d_cooRows, d_cooCols, d_cooVals, d_csrRows, CSRT);

    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooRows, d_cooRows, nnz * sizeof(int64_t), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooCols, d_cooCols, nnz * sizeof(int64_t), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooVals, d_cooVals, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_csrRows, d_csrRows, (m + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int64_t csrPtr_sol[m + 1] = {0, 2, 4, 7, 7};
    for (int i = 0; i < m + 1; ++i) {
        EXPECT_EQ(csrPtr_sol[i], h_csrRows[i]);
    }

    delete[] h_csrRows;
    util::CHECK_CUDART_ERROR(cudaFree(d_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooVals));
}

TEST(ThrustSpOps, coo2csr_2) {
    // COO format data
    int64_t m = 6;  // matrix size
    int64_t n = 5;
    int64_t nnz = 6;  // number of non-zeros
    
    // COO arrays
    int64_t h_cooRows[] = {1, 2, 2, 3, 3, 5};
    int64_t h_cooCols[] = {1, 3, 4, 0, 1, 3};
    double h_cooVals[] = {2.0, -4.0, 5.0, -1.0, 3.0, 10.0};

    // Host CSR pointer array
    int64_t* h_csrRows = new int64_t[m + 1];

    // Device arrays for COO and CSR
    int64_t *d_cooRows, *d_cooCols, *d_csrRows;
    double *d_cooVals;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooRows, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooCols, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooVals, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRows, (m + 1) * sizeof(int64_t)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooVals, h_cooVals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    cud_coo2cs(m, n, nnz, d_cooRows, d_cooCols, d_cooVals, d_csrRows, CSRT);

    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooRows, d_cooRows, nnz * sizeof(int64_t), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooCols, d_cooCols, nnz * sizeof(int64_t), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooVals, d_cooVals, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_csrRows, d_csrRows, (m + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int64_t csrPtr_sol[m + 1] = {0, 0, 1, 3, 5, 5, 6};
    int64_t cscIdx_sol[nnz] = {1, 3, 4, 0, 1, 3};
    for (int i = 0; i < m + 1; ++i) {
        EXPECT_EQ(csrPtr_sol[i], h_csrRows[i]);
    }
    for (int i = 0; i < nnz; ++i) {
        EXPECT_EQ(cscIdx_sol[i], h_cooCols[i]);
    }

    delete[] h_csrRows;
    util::CHECK_CUDART_ERROR(cudaFree(d_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooVals));
}

TEST(ThrustSpOps, coo2csc_1) {
    // COO format data
    int64_t m = 6;  // matrix size
    int64_t n = 5;
    int64_t nnz = 6;  // number of non-zeros
    
    // COO arrays
    int64_t h_cooRows[] = {1, 2, 2, 3, 3, 5};
    int64_t h_cooCols[] = {1, 3, 4, 0, 1, 3};
    double h_cooVals[] = {2.0, -4.0, 5.0, -1.0, 3.0, 10.0};

    // Host CSR pointer array
    int64_t* h_cscCols = new int64_t[n + 1];

    // Device arrays for COO and CSR
    int64_t *d_cooRows, *d_cooCols, *d_cscCols;
    double *d_cooVals;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooRows, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooCols, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooVals, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cscCols, (n + 1) * sizeof(int64_t)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooVals, h_cooVals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    cud_coo2cs(m, n, nnz, d_cooRows, d_cooCols, d_cooVals, d_cscCols, CSCT);

    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooRows, d_cooRows, nnz * sizeof(int64_t), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooCols, d_cooCols, nnz * sizeof(int64_t), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cooVals, d_cooVals, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cscCols, d_cscCols, (n + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));   

    int64_t cscPtr_sol[n + 1] = {0, 1, 3, 3, 5, 6};
    int64_t cscIdx_sol[nnz] = {3, 1, 3, 2, 5, 2};
    for (int i = 0; i < n + 1; ++i) {
        EXPECT_EQ(cscPtr_sol[i], h_cscCols[i]);
    }
    for (int i = 0; i < nnz; ++i) {
        EXPECT_EQ(cscIdx_sol[i], h_cooRows[i]);
    }

    delete[] h_cscCols;
    util::CHECK_CUDART_ERROR(cudaFree(d_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_cscCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooVals));
}

TEST(CuSparse, trsv_1)
{
    // Initialize CUSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Matrix dimensions 
    long long n = 4;
    long long nnz = 7;
    float alpha = 1.0f;
    
    // Host arrays
    long long h_csrRowPtr[5] = {0, 1, 3, 5, 7};  // CSR row pointers
    long long h_csrColInd[7] = {0, 0, 1, 0, 2, 0, 3};  // Column indices  
    float h_csrVal[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};  // Values
    float h_b[] = {1.0, 2.0, 3.0, 4.0};  // RHS vector
    float h_x[4];  // Solution vector
    float h_sol[] = {1.0f, 0.0f, -0.2f, -2.0f/7.0f};

    // Device arrays
    long long *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_b;

    // Allocate device memory
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRowPtr, (n + 1) * sizeof(long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrColInd, nnz * sizeof(long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrVal, nnz * sizeof(float)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_x, n * sizeof(float)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_b, n * sizeof(float)));

    // Copy data to device
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (n + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&matA, n, n, nnz,
                        d_csrRowPtr, d_csrColInd, d_csrVal,
                        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    cusparseDnVecDescr_t vecX, vecB;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F));
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_32F));

    // Create SpSV descriptor
    cusparseSpSVDescr_t spsvDescr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_createDescr(&spsvDescr));

    // Analyze SpSV
    size_t bufferSize;
    void* buffer = nullptr;
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecB, vecX, CUDA_R_32F,
                            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                            &bufferSize));
    util::CHECK_CUDART_ERROR(cudaMalloc(&buffer, bufferSize));

    // Analysis phase
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecB, vecX, CUDA_R_32F,
                            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                            buffer));

    // Solve phase
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecB, vecX, CUDA_R_32F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));

    // Copy result back to host
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verification
    for(int i = 0; i < n; i++) 
        EXPECT_NEAR(h_x[i], h_sol[i], 1E-6);

    // Cleanup
    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_destroyDescr(spsvDescr));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecB));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));
    
    util::CHECK_CUDART_ERROR(cudaFree(buffer));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrRowPtr));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrColInd));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrVal));
    util::CHECK_CUDART_ERROR(cudaFree(d_x));
    util::CHECK_CUDART_ERROR(cudaFree(d_b));
}

TEST(CuSparse, trsv_2) 
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    long long n = 4;
    long long nnz = 7;
    double alpha = 1.0;

    // Upper triangular matrix example
    long long h_csrRowPtr[] = {0, 3, 5, 6, 7};
    long long h_csrColInd[] = {0, 1, 2, 1, 2, 2, 3};
    double h_csrVal[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double h_b[] = {1.0, 2.0, 3.0, 4.0};
    double h_x[4];
    double h_sol[] = {-0.25, -0.125, 0.5, 4.0/7.0};

    int64_t *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal, *d_x, *d_b;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRowPtr, (n + 1) * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrColInd, nnz * sizeof(int64_t)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrVal, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_x, n * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_b, n * sizeof(double)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (n + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));

    cusparse_uptrsv_csr_64(n, n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_b, d_x, alpha);

    util::CHECK_CUDART_ERROR(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Verification
    for(int i = 0; i < n; i++)
        EXPECT_NEAR(h_x[i], h_sol[i], 1E-13);

    util::CHECK_CUDART_ERROR(cudaFree(d_csrRowPtr));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrColInd));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrVal));
    util::CHECK_CUDART_ERROR(cudaFree(d_x));
    util::CHECK_CUDART_ERROR(cudaFree(d_b));
}

TEST(CuSparse, trsv_3)
{
    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    int n = 5;
    int nnz = 13;
    double alpha = 1.0;

    // 5x5 upper triangular matrix
    int h_csrRowPtr[] = {0, 5, 8, 10, 12, 13};
    int h_csrColInd[] = {0, 1, 2, 3, 4,  // row 0
                            1, 2, 3,     // row 1
                            2, 3,        // row 2
                            3, 4,        // row 3
                            4};          // row 4
    double h_csrVal[] = {1.0, 2.0, 3.0, 4.0, 5.0,     // row 0
                              6.0, 7.0, 8.0,          // row 1
                                   9.0, 10.0,         // row 2
                                        11.0, 12.0,   // row 3
                                              13.0};  // row 4
    double h_b[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double h_x[5];

    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal, *d_x, *d_b;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRowPtr, (n + 1) * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrColInd, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrVal, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_x, n * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_b, n * sizeof(double)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matA;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&matA, n, n, nnz,
                        d_csrRowPtr, d_csrColInd, d_csrVal,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_UPPER;
    util::CHECK_CUSPARSE_ERROR(cusparseSpMatSetAttribute(matA, 
                                CUSPARSE_SPMAT_FILL_MODE,
                                &fillMode,
                                sizeof(cusparseFillMode_t)));

    cusparseDnVecDescr_t vecX, vecB;
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_64F));

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

    util::CHECK_CUDART_ERROR(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
                                        
    // Verification
    // ...

    util::CHECK_CUSPARSE_ERROR(cusparseSpSV_destroyDescr(spsvDescr));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecB));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));
    
    util::CHECK_CUDART_ERROR(cudaFree(buffer));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrRowPtr));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrColInd));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrVal));
    util::CHECK_CUDART_ERROR(cudaFree(d_x));
    util::CHECK_CUDART_ERROR(cudaFree(d_b));
} 

TEST(CuSparse, format_convert_1)
{
    cusparseHandle_t handle;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    // COO format data
    int n = 4;  // matrix size
    int nnz = 6;  // number of non-zeros
    
    // COO arrays
    int h_cooRows[] = {0, 0, 1, 2, 2, 3};
    int h_cooCols[] = {0, 1, 1, 2, 3, 3};
    double h_cooVals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Device arrays for COO and CSR
    int *d_cooRows, *d_cooCols, *d_csrRows;
    double *d_cooVals;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooRows, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooCols, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooVals, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRows, (n + 1) * sizeof(int)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooVals, h_cooVals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Convert COO to CSR
    util::CHECK_CUSPARSE_ERROR(cusparseXcoo2csr(handle, d_cooRows, nnz, n,
                        d_csrRows, CUSPARSE_INDEX_BASE_ZERO));

    int* h_csrRows = new int[n+1];
    int h_sol[] = {0, 2, 3, 5, 6};
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_csrRows, d_csrRows, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i=0; i<n+1; ++i)
        EXPECT_EQ(h_sol[i], h_csrRows[i]);

    // Clean up
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooVals));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrRows));
}

TEST(CuSparse, format_convert_2)
{
    // COO format data
    int m = 4;  // matrix size
    int n = 5;
    int nnz = 7;  // number of non-zeros
    
    // COO arrays
    int h_cooRows[] = {1, 0, 2, 0, 3, 2, 1};
    int h_cooCols[] = {1, 1, 2, 0, 3, 3, 4};
    double h_cooVals[] = {3.0, 2.0, 4.0, 1.0, 6.0, 5.0, 7.0};

    // Device arrays for COO and CSR
    int *d_cooRows, *d_cooCols, *d_csrRows;
    double *d_cooVals;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooRows, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooCols, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooVals, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_csrRows, (m + 1) * sizeof(int)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooVals, h_cooVals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Convert COO to CSR
    cusparse_dcoo2csr(m, n, nnz, d_cooRows, d_cooCols, d_cooVals, d_csrRows);

    int* h_csrRows = new int[m+1];
    int h_sol[] = {0, 2, 4, 6, 7};
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_csrRows, d_csrRows, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i=0; i<m+1; ++i)
        EXPECT_EQ(h_sol[i], h_csrRows[i]);

    // Clean up
    util::CHECK_CUDART_ERROR(cudaFree(d_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooVals));
    util::CHECK_CUDART_ERROR(cudaFree(d_csrRows));
}

TEST(CuSparse, format_convert_3)
{
    // COO format data
    int m = 4;  // matrix size
    int n = 5;
    int nnz = 7;  // number of non-zeros
    
    // COO arrays
    int h_cooRows[] = {1, 2, 0, 2, 3, 0, 1};
    int h_cooCols[] = {1, 2, 1, 3, 3, 0, 4};
    double h_cooVals[] = {3.0, 4.0, 2.0, 5.0, 6.0, 1.0, 7.0};

    // Device arrays for COO and CSR
    int *d_cooRows, *d_cooCols, *d_cscCols;
    double *d_cooVals;

    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooRows, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooCols, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cooVals, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_cscCols, (n + 1) * sizeof(int)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_cooVals, h_cooVals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Convert COO to CSR
    cusparse_dcoo2csc(m, n, nnz, d_cooRows, d_cooCols, d_cooVals, d_cscCols);

    int* h_cscCols = new int[n+1];
    int h_sol[] = {0, 1, 3, 4, 6, 7};
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_cscCols, d_cscCols, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i=0; i<n+1; ++i)
        EXPECT_EQ(h_sol[i], h_cscCols[i]);

    // Clean up
    util::CHECK_CUDART_ERROR(cudaFree(d_cooRows));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooCols));
    util::CHECK_CUDART_ERROR(cudaFree(d_cooVals));
    util::CHECK_CUDART_ERROR(cudaFree(d_cscCols));
}

TEST(CuSparse, COO_order_1)
{
    int    num_rows     = 4;
    int    num_columns  = 4;
    int    nnz          = 11;
    int    h_rows[]    = {3, 2, 0, 3, 0, 4, 1, 0, 4, 2, 2};   // unsorted
    int    h_columns[] = {1, 0, 0, 3, 2, 2, 1, 3, 1, 2, 3};   // unsorted
    double h_values[]  = {8.0, 5.0, 1.0, 9.0, 2.0, 11.0, 4.0, 3.0, 10.0, 6.0, 7.0};   // unsorted
    double h_values_sorted[11];  // nnz
    int    h_permutation[11];    // nnz
    int    h_rows_ref[]    = {0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4}; // sorted
    int    h_columns_ref[] = {0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2}; // sorted
    double h_values_ref[]  = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                              10.0, 11.0};                      // sorted
    int    h_permutation_ref[] = {2, 4, 7, 6, 1, 9, 10, 0, 3, 8, 5};
    // sort(h_coo_values)[i] = h_coo_values[h_permutation_ref[i]]
    
    // Device memory management
    int    *d_rows, *d_columns, *d_permutation;
    double *d_values, *d_values_sorted;
    void   *d_buffer;
    size_t bufferSize;
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_rows, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_columns, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_values,        nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_values_sorted, nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_permutation,   nnz * sizeof(int)));

    util::CHECK_CUDART_ERROR(cudaMemcpy(d_rows, h_rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    util::CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));
    // Create sparse vector for the permutation
    util::CHECK_CUSPARSE_ERROR(cusparseCreateSpVec(&vec_permutation, nnz, nnz,
                                        d_permutation, d_values_sorted,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    // Create dense vector for wrapping the original coo values
    util::CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vec_values, nnz, d_values, CUDA_R_64F));

    // Query working space of COO sort
    util::CHECK_CUSPARSE_ERROR(cusparseXcoosort_bufferSizeExt(handle, num_rows,
                                    num_columns, nnz, d_rows,
                                    d_columns, &bufferSize));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_buffer, bufferSize));

    // Setup permutation vector to identity
    util::CHECK_CUSPARSE_ERROR(cusparseCreateIdentityPermutation(handle, nnz, d_permutation));
    util::CHECK_CUSPARSE_ERROR(cusparseXcoosortByRow(handle, num_rows, num_columns, nnz, d_rows, d_columns, d_permutation, d_buffer));
    util::CHECK_CUSPARSE_ERROR(cusparseGather(handle, vec_values, vec_permutation));
    
    // destroy matrix/vector descriptors
    util::CHECK_CUSPARSE_ERROR(cusparseDestroySpVec(vec_permutation));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vec_values));
    util::CHECK_CUSPARSE_ERROR(cusparseDestroy(handle));

    // device result check
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_rows, d_rows, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_columns, d_columns, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_values_sorted, d_values_sorted, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_permutation, d_permutation, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (h_rows[i]          != h_rows_ref[i]    ||
            h_columns[i]       != h_columns_ref[i] ||
            h_values_sorted[i] != h_values_ref[i]  ||
            h_permutation[i]   != h_permutation_ref[i]) {
            correct = 0;
            break;
        }
    }
    
    EXPECT_TRUE(correct);

    // device memory deallocation
    util::CHECK_CUDART_ERROR(cudaFree(d_rows));
    util::CHECK_CUDART_ERROR(cudaFree(d_columns));
    util::CHECK_CUDART_ERROR(cudaFree(d_permutation));
    util::CHECK_CUDART_ERROR(cudaFree(d_values));
    util::CHECK_CUDART_ERROR(cudaFree(d_values_sorted));
    util::CHECK_CUDART_ERROR(cudaFree(d_buffer));
}

TEST(CuSparse, COO_order_2)
{
    int    num_rows     = 4;
    int    num_columns  = 4;
    int    nnz          = 11;
    int    h_rows[]    = {3, 2, 0, 3, 0, 4, 1, 0, 4, 2, 2};   // unsorted
    int    h_columns[] = {1, 0, 0, 3, 2, 2, 1, 3, 1, 2, 3};   // unsorted
    double h_values[]  = {-8.0, 5.0, 12.0, -2.0, 2.0, 11.0, 45.0, 3.0, 10.0, 6.0, 7.0};   // unsorted
    double h_values_sorted[11];  // nnz
    int    h_permutation[11];    // nnz
    int    h_rows_ref[]    = {0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4}; // sorted
    int    h_columns_ref[] = {0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2}; // sorted
    double h_values_ref[]  = {12.0, 2.0, 3.0, 45.0, 5.0, 6.0, 7.0, -8.0, -2.0,
                              10.0, 11.0};                      // sorted
    int    h_permutation_ref[] = {2, 4, 7, 6, 1, 9, 10, 0, 3, 8, 5};

    // Device memory management
    int    *d_rows, *d_columns;
    double *d_values;
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_rows, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_columns, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_values,        nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_rows, h_rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // COO sort
    cusparse_dcooRowSort(num_rows, num_columns, nnz, d_rows, d_columns, d_values);

    // device result check
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_rows, d_rows, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_columns, d_columns, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_values, d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (h_rows[i] != h_rows_ref[i]    ||
            h_columns[i] != h_columns_ref[i] ||
            h_values[i] != h_values_ref[i]) {
            correct = 0;
            break;
        }
    }
    EXPECT_TRUE(correct);

    // device memory deallocation
    util::CHECK_CUDART_ERROR(cudaFree(d_rows));
    util::CHECK_CUDART_ERROR(cudaFree(d_columns));
    util::CHECK_CUDART_ERROR(cudaFree(d_values));
}

TEST(CuSparse, COO_order_3)
{
    int    num_rows     = 4;
    int    num_columns  = 4;
    int    nnz          = 11;
    int    h_columns[] = {1,   0,   0,   3,   2,   2,    1,   3,   1,    2,   3};    // unsorted
    int    h_rows[]    = {3,   2,   0,   3,   0,   4,    1,   0,   4,    2,   2};    // unsorted
    double h_values[]  = {8.0, 5.0, 1.0, 9.0, 2.0, 11.0, 4.0, 3.0, 10.0, 6.0, 7.0};  // unsorted
    double h_values_sorted[11];  // nnz
    int    h_permutation[11];    // nnz    
    int    h_columns_ref[] = {0,   0,   1,   1,   1,    2,   2,   2,    3,   3,   3};    // sorted
    int    h_rows_ref[]    = {0,   2,   1,   3,   4,    0,   2,   4,    0,   2,   3};    // sorted
    double h_values_ref[]  = {1.0, 5.0, 4.0, 8.0, 10.0, 2.0, 6.0, 11.0, 3.0, 7.0, 9.0};  // sorted
    
    // Device memory management
    int    *d_rows, *d_columns;
    double *d_values;
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_rows, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_columns, nnz * sizeof(int)));
    util::CHECK_CUDART_ERROR(cudaMalloc((void**) &d_values,        nnz * sizeof(double)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_rows, h_rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // COO sort
    cusparse_dcooColSort(num_rows, num_columns, nnz, d_rows, d_columns, d_values);

    // device result check
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_rows, d_rows, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_columns, d_columns, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(h_values, d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (h_rows[i] != h_rows_ref[i]    ||
            h_columns[i] != h_columns_ref[i] ||
            h_values[i] != h_values_ref[i]) {
            correct = 0;
            break;
        }
    }
    EXPECT_TRUE(correct);

    // device memory deallocation
    util::CHECK_CUDART_ERROR(cudaFree(d_rows));
    util::CHECK_CUDART_ERROR(cudaFree(d_columns));
    util::CHECK_CUDART_ERROR(cudaFree(d_values));
}