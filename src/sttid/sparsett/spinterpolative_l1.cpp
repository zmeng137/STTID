#include "spmatrix.h"
#include "structures.h"
#include "util.h"
#include "mkl_kernel.h"

decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU_l1(COOMatrix_l2<double> const M_, double const cutoff, 
                        double const spthres, long long const maxdim, bool const isFullReturn)
{
    // Dimension argument check
    assertm(maxdim > 0, "maxdim must be positive!");
    std::cout << "Sparse partial rank revealing LDU decomposition l1 (double CPU) starts.\n";
    util::Timer timer("PRRLDU l1 (CPU)");

    // Initialize maximum truncation dimension k and permutations
    long long Nr = M_.rows;
    long long Nc = M_.cols;
    long long k = std::min(Nr, Nc);
    long long* rps = new long long[Nr];
    long long* cps = new long long[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);

    COOMatrix_l2<double> M(M_); // Copy input M_ to a M
    double inf_error = 0.0;     // Inference error 
    long long s = 0;              // Iteration number
    bool denseFlag = false;     // Dense/Sparse switch flag

    // Resultset initialization;
    // Here we allocate memory for d prior to other objects, since we can assign values to d at the first phase
    decompRes::SparsePrrlduRes<double> resultSet; 
    resultSet.d = new double[k];  // To be modified (narrow down)

    // Sparse-style computation 
    // A question: Do we want to sort COO every time?
    // One thing to verify: how much sparsity we lose during this outer-product iteration?
    std::cout << "PRRLDU - First Phase: Outer-product iteration starts.\n";
    while (s < k) {
        // Sparse -> Dense criteria
        util::Timer timer("PRRLDU (CPU) - Sparse phase");
        auto iter_start_time = std::chrono::high_resolution_clock::now();
        double density = double(M.nnz_count) / Nr / Nc;
        std::cout << "(Nr, Nc) = (" << Nr << ", " << Nc << "), nnz = " << M.nnz_count << ", Matrix density = " << density;
        if (density > spthres) {
            denseFlag = true;
            break;
        }
        
        // A1: Partial M, Mabs = abs=(M[s:,s:]), max value of Mabs        
        double Mabs_max = 0.0;
        double Mdenom;
        long long max_idx;
        long long nnz = M.nnz_count;
        {util::Timer timer("PRRLDU (CPU) - Sparse phase A1");
        for (long long i = 0; i < nnz; ++i) {
            if (M.row_indices[i] >= s && M.col_indices[i] >= s) {
                double Mabs = std::abs(M.values[i]);
                if (Mabs > Mabs_max) {
                    Mabs_max = Mabs;
                    Mdenom = M.values[i];
                    max_idx = i;
                }
            }                
        }}
        // termination condition
        if (Mabs_max < cutoff) {
            inf_error = Mabs_max;
            break;
        }

        // Update diagonal entries
        resultSet.d[s] = Mdenom; 

        // A2: piv, swap rows and columns  // BENCHMARK COO VS CSR (Later)
        long long piv_r = M.row_indices[max_idx];
        long long piv_c = M.col_indices[max_idx];
        {util::Timer timer("PRRLDU (CPU) - Sparse phase A2");
        for (long long i = 0; i < nnz; ++i) {
            if (M.row_indices[i] == s)
                M.row_indices[i] = piv_r;
            else if (M.row_indices[i] == piv_r)
                M.row_indices[i] = s;
            if (M.col_indices[i] == s)
                M.col_indices[i] = piv_c;
            else if (M.col_indices[i] == piv_c)
                M.col_indices[i] = s;
        }}

        // A3: Sub-matrix update by outer-product
        {util::Timer timer("PRRLDU (CPU) - Sparse phase A3");
        if (s < k - 1) {
            for (long long i = 0; i < nnz; ++i) {
                if (M.row_indices[i] == s && M.col_indices[i] > s) {
                    for (long long j = 0; j < nnz; ++j) {                      
                        if (M.col_indices[j] == s && M.row_indices[j] > s) {
                            double outprod = M.values[j] * M.values[i] / Mdenom;
                            M.addUpdate(M.row_indices[j], M.col_indices[i], -1.0 * outprod);
                        }
                    }
                }        
            }
        }}

        long long temp;
        temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
        temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;
        s += 1;

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end_time - iter_start_time);
        double runtime_ms = duration.count();
        std::cout << ", rt(ns)=" << runtime_ms << std::endl;
    }

    // Dense-style computation 
    double* M_full;
    if (denseFlag) {
        std::cout << "Dense computation starts.\n";
        util::Timer timer("PRRLDU (CPU) - Dense phase");
        M_full = M.todense();
        M.explicit_destroy();   // NOTE HERE!
        denseLU_cpukernel(M_full, Nr, Nc, s, k, cutoff, inf_error, rps, cps, resultSet);
        std::cout << "Dense computation ends.\n"; 
    }
    std::cout << "PRRLDU - Outer-product iteration ends.\n";

    // Result set
    long long rank = s;  // Detected matrix rank    
    long long output_rank = std::min(maxdim, rank);
    resultSet.rank = rank;  // Detected real rank
    resultSet.output_rank = output_rank;   // Output (truncated) rank
    resultSet.inf_error = inf_error;       // Inference error
    resultSet.isSparseRes = !denseFlag;    // Dense or Sparse result
    resultSet.isFullReturn = isFullReturn; // Full or non-full return

    // Create inverse permutations
    resultSet.col_perm_inv = new long long[Nc]{0};
    resultSet.row_perm_inv = new long long[Nr]{0};
    for (long long i = 0; i < Nr; ++i)
        resultSet.row_perm_inv[rps[i]] = i;
    for (long long j = 0; j < Nc; ++j)
        resultSet.col_perm_inv[cps[j]] = j;
    delete[] rps;
    delete[] cps;

    std::cout << "PRRLDU - Second Phase: L/U update starts.\n";
    // Whether dense LU factors or not 
    if (denseFlag) {   
        // Whether return all things or not
        if (isFullReturn) {
            // Memory allocation (full return -> return L, U, D)
            resultSet.dense_L = new double[Nr * output_rank]{0.0};
            resultSet.dense_U = new double[output_rank * Nc]{0.0};
            double* L = resultSet.dense_L;
            double* U = resultSet.dense_U;
            
            // Diagonal entries
            for (long long i = 0; i < output_rank; ++i)
                L[i * output_rank + i] = 1.0;
            for (long long i = 0; i < output_rank; ++i)
                U[i * Nc + i] = 1.0;

            // Rank-revealing Guassian elimination
            for (long long ss = 0; ss < output_rank; ++ss) {
                double P = resultSet.d[ss];
                // pivoted col
                for (long long i = ss + 1; i < Nr; ++i)
                    L[i * output_rank + ss] = M_full[i * Nc + ss] / P;
                // pivoted row
                for (long long j = ss + 1; j < Nc; ++j)
                    U[ss * Nc + j] = M_full[ss * Nc + j] / P;
                
            }
        } else {
            // Memory allocation (no full return -> return U, D)
            resultSet.dense_U = new double[output_rank * Nc]{0.0};
            double* U = resultSet.dense_U;

            // Diagonal entries
            for (long long i = 0; i < output_rank; ++i)
                U[i * Nc + i] = 1.0;

            // Rank-revealing Guassian elimination
            for (long long ss = 0; ss < output_rank; ++ss) {
                double P = resultSet.d[ss];                
                // pivoted row
                for (long long j = ss + 1; j < Nc; ++j)
                    U[ss * Nc + j] = M_full[ss * Nc + j] / P;
                
            }
        }
        // Release M_full for dense case before returning the result set
        delete[] M_full;
    } else {
        // Whether return all things or not
        if (isFullReturn) {
            // Memory allocation (full return -> return L, U, D)
            resultSet.sparse_L.reconst(Nr, output_rank);
            resultSet.sparse_U.reconst(output_rank, Nc);

            // Diagonal entries
            for (long long i = 0; i < output_rank; ++i)
                resultSet.sparse_L.add_element(i, i, 1.0);
            for (long long i = 0; i < output_rank; ++i)
                resultSet.sparse_U.add_element(i, i, 1.0);

            // Guassian elimination (New version)
            for (long long i = 0; i < M.nnz_count; ++i) {
                long long ri = M.row_indices[i];
                long long ci = M.col_indices[i];
                double val = M.values[i];
                if ((ci < resultSet.output_rank) && (ci < ri)) 
                    resultSet.sparse_L.add_element(ri, ci, val / resultSet.d[ci]);
                if ((ri < resultSet.output_rank) && (ri < ci)) 
                    resultSet.sparse_U.add_element(ri, ci, val / resultSet.d[ri]);
            }
        } else {
            // Memory allocation (no full return -> return U11, B)
            resultSet.sparse_U11.reconst(output_rank, output_rank);
            resultSet.sparse_B.reconst(output_rank, Nc - output_rank);

            // Diagonal entries
            for (long long i = 0; i < output_rank; ++i){
                resultSet.sparse_U11.add_element(i, i, 1.0);
            }
                
            // Guassian elimination (New version)
            for (long long i = 0; i < M.nnz_count; ++i) {
                long long ri = M.row_indices[i];
                long long ci = M.col_indices[i];
                double val = M.values[i];
                if ((ri < resultSet.output_rank) && (ri < ci)) {
                    if (ci < output_rank)
                        resultSet.sparse_U11.add_element(ri, ci, val / resultSet.d[ri]);
                    else 
                        resultSet.sparse_B.add_element(ri, ci - output_rank, val / resultSet.d[ri]);
                }
            }
        }
        // No need to release M_full for dense case before returning the result set
    }
    std::cout << "PRRLDU - L/U update ends.\n";
    std::cout << "Sparse partial rank revealing LDU decomposition l1 (double CPU) ends.\n";
    return resultSet;
}

decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU_l1(COOMatrix_l2<double> const M, double const cutoff, 
                        double const spthres, long long const maxdim)
{   
    std::cout << "Sparse interpolative decomposition l1 (double CPU) starts.\n";
    util::Timer timer("Sparse Interp Decomp (CPU)");

    // Result set initialization & PRRLDU decomposition
    decompRes::SparseInterpRes<double> idResult;

    // Partial rank-revealing LDU: cutoff / spthres / maxdim are controlled by input arguments of interpolative function
    // isFullReturn for prrldu function is set to FALSE by default so far
    bool isFullReturn_prrldu = false;      
    auto prrlduResult = dSparse_PartialRRLDU_CPU_l1(M, cutoff, spthres, maxdim, isFullReturn_prrldu);

    // LU reconstruction verification
    /*if (0) {
        // L * d * U = pivoted M
        long long rank = prrlduResult.rank;
        long long output_rank = prrlduResult.output_rank;
        double* reconM = new double[M.rows * M.cols]{0.0};
        double* L = prrlduResult.dense_L;
        double* U = prrlduResult.dense_U;
        double* d = prrlduResult.d;
        long long* row_perm_inv = prrlduResult.row_perm_inv;
        long long* col_perm_inv = prrlduResult.col_perm_inv;
        for (int i = 0; i < M.rows; ++i) {
            for (int j = 0; j < output_rank; ++j) 
                L[i * output_rank + j] = L[i * output_rank + j] * d[j];
            for (int j = 0; j < M.cols; ++j)
                for (int k = 0; k < output_rank; ++k)
                    reconM[i * M.cols + j] += L[i * output_rank + k] * U[k * M.cols + j];
        }
        // Reverse permutation
        double max_error = 0.0;
        double* M_full = M.todense();
        for (int i = 0; i < M.rows; ++i) 
            for (int j = 0; j < M.cols; ++j) {
                double recover = reconM[row_perm_inv[i] * M.cols + col_perm_inv[j]];
                max_error = std::max(max_error, std::abs(recover - M_full[i * M.cols + j]));
            }
        std::cout << "MAXERROR LU: " << max_error << std::endl;
        delete[] M_full;
        delete[] reconM;
    }*/

    // Rank detection
    idResult.rank = prrlduResult.rank;
    idResult.output_rank = prrlduResult.output_rank;
    
    long long output_rank = prrlduResult.output_rank;
    long long Nr = M.rows;
    long long Nc = M.cols;

    // Get pivot columns (CPU part)
    idResult.pivot_cols = new long long[Nc];
    for (long long i = 0; i < Nc; ++i) {
        long long idx;
        for (long long j = 0; j < Nc; ++j) {
            if (prrlduResult.col_perm_inv[j] == i) {
                idx = j;
                break;
            }     
        }
        idResult.pivot_cols[i] = idx;
    }

    // Allocate memory for interpolative coefficients
    if (output_rank * (Nc - output_rank) != 0) {
        idResult.interp_coeff = new double[output_rank * (Nc - output_rank)]{0.0};
    } else {
        idResult.interp_coeff = nullptr;
    }
            
    // Interpolation coefficients
    if (prrlduResult.isSparseRes) 
    {        
        // Sparse U -> Sparse interpolation
        util::Timer timer("Interp-coeff Comp (Sparse)");
        mkl_trsv_idkernel(idResult, prrlduResult, output_rank, Nc);   
    }
    else {
        // Dense U -> Dense interpolation
        util::Timer timer("Interp-coeff Comp (Dense)");
        double* U11 = new double[output_rank * output_rank]{0.0};
        double* b = new double[output_rank]{0.0};
        //util::PrintMatWindow(prrlduResult.dense_U, output_rank, Nc, {0, output_rank-1}, {0, Nc-1});

        // Dense triangular solver
        denseTRSV_interp(prrlduResult, U11, b, Nc, output_rank, idResult);

        // Memory release
        delete[] b;

        // Memory release
        delete[] b;
        delete[] U11;
    }   

    // Memory release
    prrlduResult.freeSpLduRes();

    std::cout << "Sparse interpolative decomposition l1 (double CPU) ends.\n";
    return idResult;
}

COOMatrix_l2<double> dcoeffZReconCPU(double* coeffMatrix, long long* pivot_col, long long rank, long long col)
{
    COOMatrix_l2<double> Z(rank, col);
    
    // Identity part
    for (long long i = 0; i < rank; ++i) {
        Z.add_element(i, pivot_col[i], 1.0);
    }   

    // Coefficient part
    for (long long i = rank; i < col; ++i) {   
        for (long long r = 0; r < rank; ++r) {
            double ele = coeffMatrix[r * (col - rank) + (i - rank)];
            if (std::abs(ele) > 1e-14) {
                Z.add_element(r, pivot_col[i], ele);
            }
        }
    }

    return Z;
}

