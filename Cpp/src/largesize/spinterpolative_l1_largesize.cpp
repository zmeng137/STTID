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
    std::unordered_map<long long, long long> rps;
    std::unordered_map<long long, long long> cps;

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
        std::cout << ", s=" << s << ", MabsMax=" << Mabs_max;
        if (Mabs_max < cutoff) {
            inf_error = Mabs_max;
            break;
        }

        // Update diagonal entries
        resultSet.d[s] = Mdenom; 

        // A2: piv, swap rows and columns  // BENCHMARK COO VS CSR (Later)
        long long piv_r = M.row_indices[max_idx];
        long long piv_c = M.col_indices[max_idx];
        std::cout << ", pivrc=" << piv_r << "," << piv_c; 
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
        
        auto it = rps.find(s);
        if (it != rps.end()) {
            temp = rps[s];
        } else {
            temp = s;
        }
        it = rps.find(piv_r);
        if (it != rps.end()) {
            rps[s] = rps[piv_r];
            rps[piv_r] = temp;
        } else {
            rps[s] = piv_r;
            rps[piv_r] = temp;
        }
        
        it = cps.find(s);
        if (it != cps.end()) {
            temp = cps[s];
        } else {
            temp = s;
        }
        it = cps.find(piv_c);
        if (it != cps.end()) {
            cps[s] = cps[piv_c];
            cps[piv_c] = temp;
        } else {
            cps[s] = piv_c;
            cps[piv_c] = temp;
        }
        
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
        //denseLU_cpukernel(M_full, Nr, Nc, s, k, cutoff, inf_error, rps, cps, resultSet);
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
    resultSet.piv_cols = new long long[maxdim]{0};
    resultSet.piv_rows = new long long[maxdim]{0};

    for (long long i = 0; i < maxdim; ++i) {
        auto it = cps.find(i);
        if (it != cps.end()) {
            resultSet.piv_cols[i] = cps[i];
        } else {
            resultSet.piv_cols[i] = i;
        }
    }

    for (long long i = 0; i < maxdim; ++i) {
        auto it = rps.find(i);
        if (it != rps.end()) {
            resultSet.piv_rows[i] = rps[i];
        } else {
            resultSet.piv_rows[i] = i;
        }
    }

    resultSet.rps = rps;
    resultSet.cps = cps;

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

    // Rank detection
    std::cout << "Get pivot information...\n";
    idResult.rank = prrlduResult.rank;
    idResult.output_rank = prrlduResult.output_rank;
    idResult.pivot_cols = new long long[maxdim];
    idResult.pivot_rows = new long long[maxdim];
    std::copy(prrlduResult.piv_cols, prrlduResult.piv_cols + maxdim, idResult.pivot_cols);
    std::copy(prrlduResult.piv_rows, prrlduResult.piv_rows + maxdim, idResult.pivot_rows);        
    idResult.rps = prrlduResult.rps;
    idResult.cps = prrlduResult.cps;

    long long output_rank = prrlduResult.output_rank;
    long long Nr = M.rows;
    long long Nc = M.cols;

    // Allocate memory for interpolative coefficients
    std::cout << "Get interpolative coefficients...\n";
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
        //mkl_trsv_idkernel(idResult, prrlduResult, output_rank, Nc);   
    }
    else {
        // Dense U -> Dense interpolation
        util::Timer timer("Interp-coeff Comp (Dense)");
        //double* U11 = new double[output_rank * output_rank]{0.0};
        //double* b = new double[output_rank]{0.0};
        //util::PrintMatWindow(prrlduResult.dense_U, output_rank, Nc, {0, output_rank-1}, {0, Nc-1});

        // Dense triangular solver
        //denseTRSV_interp(prrlduResult, U11, b, Nc, output_rank, idResult);

        // Memory release
        //delete[] b;

        // Memory release
        //delete[] b;
        //delete[] U11;
    }   

    // Memory release
    prrlduResult.freeSpLduRes();

    std::cout << "Sparse interpolative decomposition l1 (double CPU) ends.\n";
    return idResult;
}

