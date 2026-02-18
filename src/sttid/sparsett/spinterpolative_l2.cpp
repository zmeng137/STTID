#include "spmatrix.h"
#include "structures.h"
#include "util.h"
#include "mkl_kernel.h"

decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU_l2(COOMatrix_l2<double> const M_, double const cutoff, 
                        double const spthres, long long const maxdim, bool const isFullReturn)
{
    // Dimension argument check
    assertm(maxdim > 0, "maxdim must be positive!");
    std::cout << "Sparse partial rank revealing LDU decomposition (double CPU) starts.\n";
    util::Timer timer("PRRLDU l2 (CPU)");

    // Initialize maximum truncation dimension k and permutations
    long long Nr = M_.rows;
    long long Nc = M_.cols;
    long long k = std::min(Nr, Nc);
    long long* rps = new long long[Nr];
    long long* cps = new long long[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);

    COOMatrix_l2<double> Ms(M_);      // Copy input M_ to Ms (sub M for later Gaussian elimination)
    COOMatrix_l2<double> Mp(Nr, Nc);  // Construct Mp for recording the updated elements of M
    double inf_error = 0.0;           // Inference error 
    long long s = 0;                    // Iteration number
    bool denseFlag = false;           // Dense/Sparse switch flag

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
        double M_density = double(Ms.nnz_count + Mp.nnz_count) / Nr / Nc;
        double Ms_density = Ms.nnz_count == 0 ? 0.0 : double(Ms.nnz_count) / (Nr - s) / (Nc - s);
        double Mp_density = Mp.nnz_count == 0 ? 0.0 : double(Mp.nnz_count) / (Nr * Nc - (Nr - s) * (Nc - s));
        std::cout << "(Nr, Nc) = (" << Nr << ", " << Nc << "), total nnz(density) = " << Ms.nnz_count + Mp.nnz_count << "(" << M_density << ")";
        std::cout << ", Ms nnz(density) = " << Ms.nnz_count << "(" << Ms_density << ")";
        std::cout << ", Mp nnz(density) = " << Mp.nnz_count << "(" << Mp_density << ")";
        if (M_density > spthres) {
            denseFlag = true;
            break;
        }
        
        // A1: Partial M, Mabs = abs=(M[s:,s:]), max value of Mabs        
        double Mabs_max = 0.0;
        double Mdenom;
        long long max_idx;
        {util::Timer timer("PRRLDU (CPU) - Sparse phase A1");
        for (long long i = 0; i < Ms.nnz_count; ++i) {
            if (Ms.row_indices[i] >= s && Ms.col_indices[i] >= s) {
                double Mabs = std::abs(Ms.values[i]);
                if (Mabs > Mabs_max) {
                    Mabs_max = Mabs;
                    Mdenom = Ms.values[i];
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
        long long piv_r = Ms.row_indices[max_idx];        // pivot row
        long long piv_c = Ms.col_indices[max_idx];        // pivot column
        COOMatrix_l2<double> Mt(Nr, Nc, Ms.nnz_count);  // a temp matrix used to store unpivoted elements
        SparseVector<double> vr(Nr, 1000);              // row vector storing the pivoted row 
        SparseVector<double> vc(Nc, 1000);              // col vector storing the pivoted col
        
        // Pivot Ms
        {util::Timer timer("PRRLDU (CPU) - Sparse phase A2");
        for (long long i = 0; i < Ms.nnz_count; ++i) {
            if (Ms.row_indices[i] == s)
                Ms.row_indices[i] = piv_r;
            else if (Ms.row_indices[i] == piv_r) 
                Ms.row_indices[i] = s;
            if (Ms.col_indices[i] == s)
                Ms.col_indices[i] = piv_c;
            else if (Ms.col_indices[i] == piv_c) 
                Ms.col_indices[i] = s;            
        }

        // Pivot Mp
        for (long long i = 0; i < Mp.nnz_count; ++i) {
            if (Mp.row_indices[i] == s)
                Mp.row_indices[i] = piv_r;
            else if (Mp.row_indices[i] == piv_r) 
                Mp.row_indices[i] = s;
            if (Mp.col_indices[i] == s)
                Mp.col_indices[i] = piv_c;
            else if (Mp.col_indices[i] == piv_c) 
                Mp.col_indices[i] = s;
        }

        // Extract elements to Mt, Mp, and vectors vr, vc
        for (long long i = 0; i < Ms.nnz_count; ++i) {
            long long r_i = Ms.row_indices[i];
            long long c_i = Ms.col_indices[i];
            double v_i = Ms.values[i];
            if (r_i > s && c_i > s) {
                Mt.add_element(r_i, c_i, v_i);
            } else {
                Mp.add_element(r_i, c_i, v_i);
                if (r_i != c_i) {
                    if (r_i == s)
                    vr.set(c_i, v_i);
                if (c_i == s)
                    vc.set(r_i, v_i);
                }    
            }
        }}

        // A3: Sub-matrix update by outer-product
        // Outer product of vr and vc: Addupdate Mt = Mt - (vr o vc)
        {util::Timer timer("PRRLDU (CPU) - Sparse phase A3");
        if (s < k - 1) {
            double outprod = 0.0;
            for (long long i = 0; i < vc.nnz; ++i)
                for (long long j = 0; j < vr.nnz; ++j) {
                    outprod = vc.val[i] * vr.val[j] / Mdenom;
                    Mt.addUpdate(vc.idx[i], vr.idx[j], -1.0 * outprod);
            }
        }}

        Ms = Mt;
        long long temp;
        temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
        temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;
        s += 1;

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end_time - iter_start_time);
        double runtime_ms = duration.count();
        std::cout << ", rt(ns)=" << runtime_ms << std::endl;
    }

    // Post-processing for sparse computation
    COOMatrix_l2<double> M(Nr, Nc, Ms.nnz_count + Mp.nnz_count);  // Construct M for the entire updated matrix (M <- Mp + Ms)
    M.nnz_count = Ms.nnz_count + Mp.nnz_count;
    std::copy(Mp.row_indices, Mp.row_indices + Mp.nnz_count, M.row_indices);
    std::copy(Ms.row_indices, Ms.row_indices + Ms.nnz_count, M.row_indices + Mp.nnz_count);
    std::copy(Mp.col_indices, Mp.col_indices + Mp.nnz_count, M.col_indices);
    std::copy(Ms.col_indices, Ms.col_indices + Ms.nnz_count, M.col_indices + Mp.nnz_count);
    std::copy(Mp.values, Mp.values + Mp.nnz_count, M.values);
    std::copy(Ms.values, Ms.values + Ms.nnz_count, M.values + Mp.nnz_count);
    Mp.explicit_destroy();
    Ms.explicit_destroy();

    // Dense-style computation 
    double* M_full;  // Dense M
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
    std::cout << "Sparse partial rank revealing LDU decomposition (double CPU) ends.\n";
    return resultSet;
}

decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU_l2(COOMatrix_l2<double> const M, double const cutoff, 
                        double const spthres, long long const maxdim)
{   
    std::cout << "Sparse interpolative decomposition l2 (double CPU) starts.\n";
    util::Timer timer("Sparse Interp Decomp (CPU)");

    // Result set initialization & PRRLDU decomposition
    decompRes::SparseInterpRes<double> idResult;

    // Partial rank-revealing LDU: cutoff / spthres / maxdim are controlled by input arguments of interpolative function
    // isFullReturn for prrldu function is set to FALSE by default so far
    bool isFullReturn_prrldu = false;      
    auto prrlduResult = dSparse_PartialRRLDU_CPU_l2(M, cutoff, spthres, maxdim, isFullReturn_prrldu);

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
        mkl_trsv_idkernel(idResult, prrlduResult, output_rank, Nc);     }
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

    std::cout << "Sparse interpolative decomposition l2 (double CPU) ends.\n";
    return idResult;
}