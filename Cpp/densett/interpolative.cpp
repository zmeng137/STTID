#include "dfunctions.h"
#include "util.h"
#include <cblas.h>
#include <lapacke.h>

// Interpolative decomposition by pivoted QR
void dInterpolative_PivotedQR(double* M, int m, int n, int maxdim, 
                              double* C, double* Z, int& outdim)
{
    // Get CZ rank k
    int k = maxdim;
    
    // Pivoted (rank-revealing) QR decomposition
    double* Q = new double[m * n]{0.0};
    double* R = new double[n * n]{0.0};
    int* P = new int[n];
    int rank;
    dPivotedQR_MGS(M, m, n, Q, R, P, rank);
    k = k < rank ? k : rank;
    outdim = k;

    // R_k = R[0:k,0:k] (To be optimized)
    double* R_k = new double[k * k];
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) 
            R_k[i * k + j] = R[i * n + j];
    
    // C = M[:, cols]     TOBECONTINUED... Rank stuff...
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            C[i * k + j] = M[i * n + P[j]];

    // Solve linear systems for Z: (R_k^T * R_k) Z = C^T * M
    double* b = new double[k];
    for (int i = 0; i < n; ++i) {
        // Construct right hand side b = C^T * M[:,i]
        std::fill(b, b + k, 0.0);
        for (int j = 0; j < k; ++j) 
            for (int l = 0; l < m; ++l) 
                b[j] += C[l * k + j] * M[l * n + i];
        // Solve two triangular systems R_k/R_k^T
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, k, R_k, k, b, 1);
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, k, R_k, k, b, 1);  
        // Copy solution to Z
        for (int j = 0; j < k; ++j) 
            Z[j * n + i] = b[j];
    }

    delete[] Q;
    delete[] R;
    delete[] P;
    delete[] R_k;
    delete[] b;
    return;        
}

void dInterpolative_PrrLDU(double* M, size_t Nr, size_t Nc, size_t maxdim, double cutoff,
                           double* C, double* Z, size_t& outdim)
{
    // Partial rank-revealing LDU decomposition
    auto prrlduResult = dPartialRRLDU(M, Nr, Nc, cutoff, maxdim, 1);
    size_t k = prrlduResult.rank;
    outdim = k;

    // Extract relevant submatrices
    double* U11 = new double[k * k]{0.0};
    for (size_t i = 0; i < k; ++i)
        std::copy(prrlduResult.U + i * Nc, prrlduResult.U + i * Nc + k, U11 + i * k);

    // Compute inverse of U11 through backward-substitution solving
    double* iU11 = new double[k * k]{0.0};
    double* b = new double[k]{0.0};
    
    /*
    for (size_t i = 0; i < k; ++i) {
        util::Timer timer("trsv");
        // Right hand side b (one column of the diagonal matrix)
        std::fill(b, b + k, 0.0);
        b[i] = 1.0;
        
        // Triangular solver (BLAS)        
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, k, U11, k, b, 1);
        
        // Copy the solution to iU11 columns
        for (size_t j = 0; j < k; ++j) 
            iU11[j * k + i] = b[j];
    }
    */

    // Compute the interpolation matrix
    double* ZjJ = new double[k * Nc]{0.0};
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < Nc; ++j)
            for (size_t m = 0; m < k; ++m)
                ZjJ[i * Nc + j] += iU11[i * k + m] * prrlduResult.U[m * Nc + j];

    // Compute selected columns 
    double* CIj = new double[Nr * k]{0.0};
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < k; ++j) 
            U11[i * k + j] = prrlduResult.d[i] * U11[i * k + j];
    for (size_t i = 0; i < Nr; ++i)
        for (size_t j = 0; j < k; ++j) 
            for (size_t m = 0; m < k; ++m)
                CIj[i * k + j] += prrlduResult.L[i * k + m] * U11[m * k + j];
    
    // Apply row and column permutation to get C and Z
    for (size_t i = 0; i < Nr; ++i) {
        size_t pr = prrlduResult.row_perm_inv[i];
        std::copy(CIj + pr * k, CIj + pr * k + k, C + i * k);
    }
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < Nc; ++j) {
            size_t pc = prrlduResult.col_perm_inv[j];
            Z[i * Nc + j] = ZjJ[i * Nc + pc];
        }
                
    // Memory release
    delete[] U11;
    delete[] iU11;
    delete[] ZjJ;
    delete[] CIj;
    delete[] b;
    prrlduResult.freeLduRes();

    return;
}