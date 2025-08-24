#include "spmatrix.h"
#include "structures.h"
#include "util.h"
#include "mkl_kernel.h"

#include "spmatrix_device.h"
#include "gpu_kernel.h"

//#define KFUSION   // optimized runtime
//#define nnz_log   // output fill-ins to a file

void denseLU_gpukernel(double* M_full, const long long Nr, const long long Nc, 
    long long& s, const long long k, const double cutoff, double& inf_error, 
    long long* rps, long long* cps, decompRes::SparsePrrlduRes<double> resultSet);
    
void cusparse_trsv_idkernel_3(decompRes::SparseInterpRes<double> idResult, 
    decompRes::SparsePrrlduRes<double> prrlduResult, long long output_rank, long long Nc);     

decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_GPU_l3(COOMatrix_l2<double> const M_, double const cutoff, 
                        double const spthres, long long const maxdim, bool const isFullReturn)
{
    // Initial spell
    assertm(maxdim > 0, "maxdim must be positive!");  // Dimension argument check
    cudaSetDevice(0); 
    int gpuDevice; cudaGetDevice(&gpuDevice);
    cudaDeviceProp prop;
    util::CHECK_CUDART_ERROR(cudaGetDeviceProperties(&prop, 0));  // Get properties of device 0
    std::cout << "Sparse PRRLDU decomposition on GPU starts. ";
    std::cout << "Currently using device: " << gpuDevice << ". Device name: " << prop.name << std::endl;
    device_warmup();   // Warm up the GPU device
    util::Timer timer("PRRLDU (GPU)");

    // Initialize maximum truncation dimension k and permutations
    const long long Nr = M_.rows;
    const long long Nc = M_.cols;
    const long long k = std::min(std::min(Nr, Nc), maxdim);
    long long* rps = new long long[Nr];
    long long* cps = new long long[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);
    
    // Ms and Mp on device   
    //COOMatrix_l2<double> Ms(M_);             // Copy input on-host M_ to on-host Ms (sub M for later Gaussian elimination)
    //COOMatrix_l2<double> Mp(Nr, Nc);         // Construct on-host Mp for recording the updated elements of M
    COOMatrix_l2_device<double> d_Ms(M_);      // Copy input on-host M_ to on-device Ms (sub M for later Gaussian elimination)
    COOMatrix_l2_device<double> d_Mp(Nr, Nc);  // Construct on-device Mp for recording the updated elements of M
    
    #ifdef nnz_log
    std::ofstream M_nnz_outFile("M_nnz.txt", std::ios::app);
    std::ofstream Ms_nnz_outFile("Ms_nnz.txt", std::ios::app);
    std::ofstream Mp_nnz_outFile("Mp_nnz.txt", std::ios::app);
    #endif

    double inf_error = 0.0;  // Inference error 
    long long s = 0;         // Iteration number
    bool denseFlag = false;  // Dense/Sparse switch flag

    // Resultset initialization;
    // Here we allocate memory for d prior to other objects, since we can assign values to d at the first phase
    decompRes::SparsePrrlduRes<double> resultSet; 
    resultSet.d = new double[k];  
    
    // Sparse-style computation (One thing to verify: how much sparsity we lose during this outer-product iteration?)
    std::cout << "PRRLDU - First Phase: Outer-product iteration starts.\n";
    while (s < k) {
        // Sparse -> Dense criteria
        util::Timer timer("PRRLDU (GPU) - Sparse phase");
        auto iter_start_time = std::chrono::high_resolution_clock::now();
        double M_density = double(d_Ms.nnz_count + d_Mp.nnz_count) / Nr / Nc;
        double Ms_density = d_Ms.nnz_count == 0 ? 0.0 : double(d_Ms.nnz_count) / (Nr - s) / (Nc - s);
        double Mp_density = d_Mp.nnz_count == 0 ? 0.0 : double(d_Mp.nnz_count) / (Nr * Nc - (Nr - s) * (Nc - s));
        std::cout << "(Nr,Nc)=(" << Nr << "," << Nc << "), MNNZ=" << d_Ms.nnz_count + d_Mp.nnz_count << "(" << M_density << ")";
        std::cout << ", MsNNZ=" << d_Ms.nnz_count << "(" << Ms_density << ")";
        std::cout << ", MpNNZ=" << d_Mp.nnz_count << "(" << Mp_density << ")";
        
        #ifdef nnz_log
        M_nnz_outFile << d_Ms.nnz_count + d_Mp.nnz_count << ", " << M_density << "\n";
        Mp_nnz_outFile << d_Mp.nnz_count << ", " << Mp_density << "\n";
        Ms_nnz_outFile << d_Ms.nnz_count << ", " << Ms_density << "\n";
        #endif

        if (M_density > spthres) {
            util::CHECK_LAST_CUDART_ERROR();
            denseFlag = true;
            break;
        }
        
        //*** A1: Partial M, Mabs = abs=(M[s:,s:]), max value of Mabs ***//   
        double max_val = 0.0;
        long long max_idx;
        findMaxAbsValueCublas(d_Ms.d_values, d_Ms.nnz_count, max_idx, max_val);
        util::CHECK_LAST_CUDART_ERROR();
        double Mdenom = max_val;
        double Mabs_max = std::abs(max_val);
        
        // termination condition
        std::cout << ", s=" << s << ", MabsMax=" << Mabs_max;
        if (Mabs_max < cutoff || s > (maxdim + 1)) {
            util::CHECK_LAST_CUDART_ERROR();
            std::cout << ". Terminate!" << std::endl;
            inf_error = Mabs_max;
            break;
        }

        // Update diagonal entries
        resultSet.d[s] = Mdenom; 

        // Get pivot row and col index
        long long piv_r, piv_c;                         // pivot row and column index
        util::CHECK_CUDART_ERROR(cudaMemcpy(&piv_r, d_Ms.d_row_indices + max_idx, sizeof(long long), cudaMemcpyDeviceToHost));
        util::CHECK_CUDART_ERROR(cudaMemcpy(&piv_c, d_Ms.d_col_indices + max_idx, sizeof(long long), cudaMemcpyDeviceToHost));
        std::cout << ", pivrc=" << piv_r << "," << piv_c;
        
        // Object allocation prior to A2/A3/A4/A5 
        unsigned long long ins_cnt_flag;
        d_Mp.resize(d_Mp.nnz_count + d_Ms.nnz_count); // Resize d_Mp if necessary (This part could be more efficient. To be modified)
        SparseVector_device<double> d_vr(Nc, Nc);     // row vector storing the pivoted row 
        SparseVector_device<double> d_vc(Nr, Nr);     // col vector storing the pivoted col
        SparseVector_device<double> d_Mt_hash(Nr * Nc, d_Ms.nnz_count);  // Hash table on device

        #ifdef KFUSION
        // Fusion Kernel A2 A3 A4 A5
        ins_cnt_flag = A2345_fusion( 
            d_Ms.d_row_indices, d_Ms.d_col_indices, d_Ms.d_values, d_Ms.nnz_count, d_Ms.capacity,   
            d_Mp.d_row_indices, d_Mp.d_col_indices, d_Mp.d_values, d_Mp.nnz_count,                            
            d_Mt_hash.d_idx, d_Mt_hash.d_val, d_Mt_hash.nnz,                           
            d_vr.d_idx, d_vr.d_val, d_vr.nnz,
            d_vc.d_idx, d_vc.d_val, d_vc.nnz,
            piv_r, piv_c, Nc, Mdenom, s);
        #else
        {util::Timer timer("PRRLDU (GPU) - Sparse A2+3+4+5");
        //--- A2: Swap rows and columns ---//
        coo_pivot_gpu(d_Mp.d_row_indices, d_Mp.d_col_indices, d_Mp.nnz_count, piv_r, piv_c, s);
        coo_pivot_gpu(d_Ms.d_row_indices, d_Ms.d_col_indices, d_Ms.nnz_count, piv_r, piv_c, s);

        //--- A3: Extract elements to Mt, Mp, and vectors vr, vc ---//
        eleClassify_gpu(d_Ms.d_row_indices, d_Ms.d_col_indices, d_Ms.d_values, d_Ms.nnz_count, 
            d_Mp.d_row_indices, d_Mp.d_col_indices, d_Mp.d_values, d_Mp.nnz_count, 
            d_Mt_hash.d_idx, d_Mt_hash.d_val, d_Mt_hash.nnz, Nc, 
            d_vr.d_idx, d_vr.d_val, d_vr.nnz, d_vc.d_idx, d_vc.d_val, d_vc.nnz, s);

        //--- A4: Outer-product update of Ms(by hash table) ---//
        ins_cnt_flag = outerproduct_update_gpu_opt(
            d_Mt_hash.d_idx, d_Mt_hash.d_val, d_Mt_hash.nnz,    // In/Out: Hash table Mt
            d_vr.d_idx, d_vr.d_val, d_vr.nnz,                   // In: Sparse Vr
            d_vc.d_idx, d_vc.d_val, d_vc.nnz,                   // In: Sparse Vc
            Mdenom, Nc);
        
        //--- A5: Assign hash table to Ms for next iteration ---//
        d_Ms.reconst(Nr, Nc, d_Mt_hash.nnz);
        hash2COO_gpu(d_Ms.d_row_indices, d_Ms.d_col_indices, d_Ms.d_values, d_Ms.nnz_count,          
            d_Mt_hash.d_idx, d_Mt_hash.d_val, d_Mt_hash.nnz, Nc);}
        #endif

        long long temp;
        temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
        temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;
        s += 1;

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end_time - iter_start_time);
        double runtime_ms = duration.count();
        std::cout << ", rt(ms)=" << runtime_ms << std::endl;
    }

    util::CHECK_LAST_CUDART_ERROR();
    
    // Post-processing for sparse computation
    COOMatrix_l2<double> M(Nr, Nc, d_Ms.nnz_count + d_Mp.nnz_count);  // Construct M for the entire updated matrix (M <- Mp + Ms)
    M.nnz_count = d_Ms.nnz_count + d_Mp.nnz_count;
    
    // Copy Mp + Ms to M
    util::CHECK_CUDART_ERROR(cudaMemcpy(M.row_indices, d_Mp.d_row_indices, sizeof(long long) * d_Mp.nnz_count, cudaMemcpyDeviceToHost));  
    util::CHECK_CUDART_ERROR(cudaMemcpy(M.col_indices, d_Mp.d_col_indices, sizeof(long long) * d_Mp.nnz_count, cudaMemcpyDeviceToHost));  
    util::CHECK_CUDART_ERROR(cudaMemcpy(M.values,      d_Mp.d_values,      sizeof(double) * d_Mp.nnz_count,    cudaMemcpyDeviceToHost));  
    util::CHECK_CUDART_ERROR(cudaMemcpy(M.row_indices + d_Mp.nnz_count, d_Ms.d_row_indices, sizeof(long long) * d_Ms.nnz_count, cudaMemcpyDeviceToHost));  
    util::CHECK_CUDART_ERROR(cudaMemcpy(M.col_indices + d_Mp.nnz_count, d_Ms.d_col_indices, sizeof(long long) * d_Ms.nnz_count, cudaMemcpyDeviceToHost));  
    util::CHECK_CUDART_ERROR(cudaMemcpy(M.values + d_Mp.nnz_count,      d_Ms.d_values,      sizeof(double) * d_Ms.nnz_count,    cudaMemcpyDeviceToHost));

    // Explicitly release memory
    d_Mp.explicit_destroy();
    d_Ms.explicit_destroy();
    
    // Dense-style computation 
    double* M_full;  // Dense M
    if (denseFlag) {
        std::cout << "Dense computation starts.\n";
        util::Timer timer("PRRLDU (GPU) - Dense phase");
        M_full = M.todense();
        M.explicit_destroy();   // NOTE HERE!
        denseLU_gpukernel(M_full, Nr, Nc, s, k, cutoff, inf_error, rps, cps, resultSet);
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
        util::Timer timer("PRRLDU (GPU) - Dense ResUpdate");
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
        util::Timer timer("PRRLDU (GPU) - Sparse ResUpdate");
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
    std::cout << "Sparse partial rank revealing LDU decomposition (double GPU) ends.\n";
    return resultSet;
}

decompRes::SparseInterpRes<double>
dSparse_Interpolative_GPU_l3(COOMatrix_l2<double> const M, double const cutoff, 
                        double const spthres, long long const maxdim)
{   
    std::cout << "Sparse interpolative decomposition l3 (double GPU) starts.\n";
    util::Timer timer("Sparse Interp Decomp (GPU)");

    // Result set initialization & PRRLDU decomposition
    decompRes::SparseInterpRes<double> idResult;

    // Partial rank-revealing LDU: cutoff / spthres / maxdim are controlled by input arguments of interpolative function
    // isFullReturn for prrldu function is set to FALSE by default so far
    bool isFullReturn_prrldu = false;      
    auto prrlduResult = dSparse_PartialRRLDU_GPU_l3(M, cutoff, spthres, maxdim, isFullReturn_prrldu);

    // Rank detection
    std::cout << "Get pivot information...\n";
    idResult.rank = prrlduResult.rank;
    idResult.output_rank = prrlduResult.output_rank;
    
    long long output_rank = prrlduResult.output_rank;
    long long Nr = M.rows;
    long long Nc = M.cols;

    // Get pivot rows and columns (CPU part)
    idResult.pivot_cols = new long long[Nc];
    idResult.pivot_rows = new long long[Nr];
    long long* d_pivot_cols, *d_pivot_rows, *d_col_perm_inv, *d_row_perm_inv;
    
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_row_perm_inv, Nr * sizeof(long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_col_perm_inv, Nc * sizeof(long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_pivot_rows,   Nr * sizeof(long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_pivot_cols,   Nc * sizeof(long long)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_row_perm_inv, prrlduResult.row_perm_inv, Nr * sizeof(long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_col_perm_inv, prrlduResult.col_perm_inv, Nc * sizeof(long long), cudaMemcpyHostToDevice));
    
    perm_inv_gpu(d_pivot_rows, d_row_perm_inv, Nr);
    perm_inv_gpu(d_pivot_cols, d_col_perm_inv, Nc);
    
    util::CHECK_CUDART_ERROR(cudaMemcpy(idResult.pivot_rows, d_pivot_rows, Nr * sizeof(long long), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(idResult.pivot_cols, d_pivot_cols, Nc * sizeof(long long), cudaMemcpyDeviceToHost));          
    util::CHECK_CUDART_ERROR(cudaFree(d_row_perm_inv));
    util::CHECK_CUDART_ERROR(cudaFree(d_col_perm_inv));
    util::CHECK_CUDART_ERROR(cudaFree(d_pivot_rows));
    util::CHECK_CUDART_ERROR(cudaFree(d_pivot_cols));

    /*
    for (long long i = 0; i < Nc; ++i) {
        long long idx;
        for (long long j = 0; j < Nc; ++j) {
            if (prrlduResult.col_perm_inv[j] == i) {
                idx = j;
                break;
            }     
        }
        idResult.pivot_cols[i] = idx;
    }*/

    std::cout << "Get interpolative coefficients...\n";
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
        if (Nc != output_rank + 1)
        mkl_trsv_idkernel(idResult, prrlduResult, output_rank, Nc);    // CPU MKL kernel
        //cusparse_trsv_idkernel_3(idResult, prrlduResult, output_rank, Nc); // GPU CUSPARSE kernel
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
        delete[] U11;
    }   

    // Memory release
    prrlduResult.freeSpLduRes();

    std::cout << "Sparse interpolative decomposition l3 (double GPU) ends.\n";
    return idResult;
}