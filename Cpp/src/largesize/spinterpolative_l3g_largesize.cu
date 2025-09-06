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
    cudaSetDevice(0);  // Needed?
    int gpuDevice; cudaGetDevice(&gpuDevice);
    std::cout << "Sparse PRRLDU decomposition on GPU starts. ";
    std::cout << "Currently using device: " << gpuDevice << std::endl;
    device_warmup();   // Warm up the GPU device
    util::Timer timer("PRRLDU (GPU)");

    // Initialize maximum truncation dimension k and permutations
    const long long Nr = M_.rows;
    const long long Nc = M_.cols;
    const long long k = std::min(std::min(Nr, Nc), maxdim);
    std::unordered_map<long long, long long> rps;
    std::unordered_map<long long, long long> cps;
    
    // Ms and Mp on device   
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
        double M_density = double(d_Ms.nnz_count + d_Mp.nnz_count) / Nr / Nc;
        double Ms_density = d_Ms.nnz_count == 0 ? 0.0 : double(d_Ms.nnz_count) / (Nr - s) / (Nc - s);
        double Mp_density = d_Mp.nnz_count == 0 ? 0.0 : double(d_Mp.nnz_count) / (Nr * Nc - (Nr - s) * (Nc - s));
        std::cout << "(Nr,Nc)=(" << Nr << "," << Nc << "), totalnnz(density)=" << d_Ms.nnz_count + d_Mp.nnz_count << "(" << M_density << ")";
        std::cout << ", Ms nnz(density)=" << d_Ms.nnz_count << "(" << Ms_density << ")";
        std::cout << ", Mp nnz(density)=" << d_Mp.nnz_count << "(" << Mp_density << ")";
        
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
        std::cout << ", pivrc=" << piv_r << "," << piv_c << std::endl;
        
        // Object allocation prior to A2/A3/A4/A5 
        unsigned long long ins_cnt_flag;
        d_Mp.resize(d_Mp.nnz_count + d_Ms.nnz_count); // Resize d_Mp if necessary (This part could be more efficient. To be modified)
        SparseVector_device<double> d_vr(Nc, d_Ms.nnz_count);     // row vector storing the pivoted row 
        SparseVector_device<double> d_vc(Nr, d_Ms.nnz_count);     // col vector storing the pivoted col
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
    
  
    std::cout << "PRRLDU - Outer-product iteration ends.\n";

    // Result set
    long long rank = s;  // Detected matrix rank    
    long long output_rank = std::min(maxdim, rank);
    resultSet.rank = rank;  // Detected real rank
    resultSet.output_rank = output_rank;   // Output (truncated) rank
    resultSet.inf_error = inf_error;       // Inference error
    resultSet.isSparseRes = !denseFlag;    // Dense or Sparse result
    resultSet.isFullReturn = isFullReturn; // Full or non-full return

    // Pivots
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

    // Whether dense LU factors or not 
    std::cout << "PRRLDU - Second Phase: L/U update starts.\n";
    if (denseFlag) {   
       
    } else {
        util::Timer timer("PRRLDU (GPU) - Sparse ResUpdate");
        // Whether return all things or not
        if (isFullReturn) {
            // Memory allocation (full return -> return L, U, D)
            /*
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
            */

            // Memory re-allocation (full return -> return cross part of L, U)
            resultSet.sparse_L.reconst(output_rank, output_rank);
            resultSet.sparse_U.reconst(output_rank, output_rank);
            
            // Diagonal entries
            for (long long i = 0; i < output_rank; ++i)
                resultSet.sparse_L.add_element(i, i, resultSet.d[i]);
            for (long long i = 0; i < output_rank; ++i)
                resultSet.sparse_U.add_element(i, i, 1.0);
            
            // Guassian elimination (New version)
            for (long long i = 0; i < M.nnz_count; ++i) {
                long long ri = M.row_indices[i];
                long long ci = M.col_indices[i];
                double val = M.values[i];
                if ((ci < resultSet.output_rank) && (ri < resultSet.output_rank) && (ci < ri)) 
                    resultSet.sparse_L.add_element(ri, ci, val);
                if ((ri < resultSet.output_rank) && (ci < resultSet.output_rank) && (ri < ci)) 
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
                        double const spthres, long long const maxdim, bool const isCrossReturn)
{   
    std::cout << "Sparse interpolative decomposition l3 (double GPU) starts.\n";
    util::Timer timer("Sparse Interp Decomp (GPU)");

    // Result set initialization & PRRLDU decomposition
    decompRes::SparseInterpRes<double> idResult;

    // Partial rank-revealing LDU: cutoff / spthres / maxdim are controlled by input arguments of interpolative function
    // isFullReturn for prrldu function is set to FALSE by default so far
    bool isFullReturn_prrldu = isCrossReturn;      
    auto prrlduResult = dSparse_PartialRRLDU_GPU_l3(M, cutoff, spthres, maxdim, isFullReturn_prrldu);

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

    std::cout << "Get interpolative coefficients...\n";
    if (output_rank * (Nc - output_rank) != 0) {
        idResult.sparse_interp_coeff.reconst(output_rank, Nc - output_rank);
    }
            
    // Interpolation coefficients or CROSS inverse
    if (prrlduResult.isSparseRes) {        
        if (!isCrossReturn) {
            util::Timer timer("ID Interp-coeff (Sparse)");            
            if (Nc != output_rank + 1)
                mkl_trsv_idkernel(idResult.sparse_interp_coeff, prrlduResult, output_rank, Nc);   // CPU MKL kernel
                //cusparse_trsv_idkernel_3(idResult, prrlduResult, output_rank, Nc);                // GPU CUSPARSE kernel
        } else {
            util::Timer timer("CROSS Inverse (Sparse)");
            std::cout << "CROSS Inverse kernel has not been finalized!" << std::endl;
            //if (Nc ! = output_rank + 1)
                // todo?
        }   
    }
    else {
        // Dense U -> Dense interpolation
        util::Timer timer("Interp-coeff Comp (Dense)");
        double* U11 = new double[output_rank * output_rank]{0.0};
        double* b = new double[output_rank]{0.0};
        util::PrintMatWindow(prrlduResult.dense_U, output_rank, Nc, {0, output_rank-1}, {0, Nc-1});

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