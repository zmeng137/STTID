#include "sptensor.h"
#include "spfunctions.h"
#include "util.h"

#define ENABLE_GPU

// Define the template function TT_ID_sparse
template<typename T, size_t Order> SparseTTRes 
TT_ID_sparse(const COOTensor<T, Order>& tensor, double const cutoff, size_t const r_max,
             double const spthres, bool check_flag, bool cross_flag) 
{
    util::Timer timer("Tensor train decomp");

    auto shape = tensor.get_dimensions();  // Get the shape of the input tensor: [n1, n2, ..., nd]
    int dim = shape.size();                // Get the number of dimension d
    
    // Get the total size: n1 * n2 * ... * nd
    __int128_t nbar = 1;  // For some tensors with very large dimensions, we need 128bit int
    for (int i = 0; i < dim; ++i)
        nbar *= shape[i];

    // Reshape the input tensor to the 2D matrix
    size_t r = 1;                           // Rank
    size_t row = nbar / r / shape[dim - 1]; // Initial matrix row
    size_t col = r * shape[dim - 1];        // Initial matrix column
    
    // Tensor -- reshape --> Matrix
    COOMatrix_l2<T> W = tensor.reshape2Matl2(row, col);

    // Open a file for recording number of nonzeros for TT-factors (either one-side ID or two-side CROSS format)
    std::ofstream outFile("nnz_log.txt");
    if (!outFile.is_open()) 
        std::cerr << "Error opening file!" << std::endl;

    // Initialize a result list, start TT iteration
    SparseTTRes ResList;
    ResList.InterG.resize(dim - 2);
    ResList.InterC.resize(dim - 1);

    for (int i = dim - 1; i > 0; i--) {
        std::cout << "Tensor train iteration " << i << " starts..." << std::endl;

        // Reshape matrix (skip the first iteration)
        if (i != dim - 1) {
            row = nbar / r / shape[i];  // Matrix row
            col = r * shape[i];         // Matrix column
            W.reshape(row, col);
        }

        // Sparse interpolative decomposition
        #ifdef ENABLE_GPU
        auto idResult = dSparse_Interpolative_GPU_l3(W, cutoff, spthres, r_max, cross_flag);
        #else
        //auto idResult = dSparse_Interpolative_CPU_l3(W, cutoff, spthres, r_max, cross_flag);
        auto idResult = dSparse_Interpolative_CPU_l1(W, cutoff, spthres, r_max);
        #endif

        // There is no cutoff selection. Rank is revealed automatically by ID-PRRLU
        size_t ri = idResult.output_rank;

        // Output
        if (cross_flag) {
            // TT-core and pivot matrix
            auto row_subset = W.subrow(idResult.pivot_rows, ri);
            auto pivot_mat = row_subset.subcol(idResult.pivot_cols, ri);
            
            // TT-cores
            if (i == dim - 1) {
                outFile << "TT representation (TCI format)" << std::endl;
                ResList.EndG.reset(row_subset.nnz_count, ri, shape[i]);
                ResList.EndG.nnz_count = row_subset.nnz_count;
                std::copy(row_subset.values, row_subset.values + row_subset.nnz_count, ResList.EndG.values);
                std::copy(row_subset.row_indices, row_subset.row_indices + row_subset.nnz_count, ResList.EndG.indices[0]);
                std::copy(row_subset.col_indices, row_subset.col_indices + row_subset.nnz_count, ResList.EndG.indices[1]);
            } else {
                ResList.InterG[i-1].reset(row_subset.nnz_count, ri, shape[i], r);
                ResList.InterG[i-1].nnz_count = row_subset.nnz_count;
                std::copy(row_subset.values, row_subset.values + row_subset.nnz_count, ResList.InterG[i-1].values);
                std::copy(row_subset.row_indices, row_subset.row_indices + row_subset.nnz_count, ResList.InterG[i-1].indices[0]);
                for (size_t n = 0; n < row_subset.nnz_count; ++n) {
                    ResList.InterG[i-1].indices[1][n] = row_subset.col_indices[n] / r;
                    ResList.InterG[i-1].indices[2][n] = row_subset.col_indices[n] % r;
                }
            }
            
            // Pivot matrices
            ResList.InterC[i-1].reset(pivot_mat.nnz_count, ri, ri);
            ResList.InterC[i-1].nnz_count = pivot_mat.nnz_count;
            std::copy(pivot_mat.values, pivot_mat.values + pivot_mat.nnz_count, ResList.InterC[i-1].values);
            std::copy(pivot_mat.row_indices, pivot_mat.row_indices + pivot_mat.nnz_count, ResList.InterC[i-1].indices[0]);
            std::copy(pivot_mat.col_indices, pivot_mat.col_indices + pivot_mat.nnz_count, ResList.InterC[i-1].indices[1]);

            // Output NNZ info
            outFile << "Mode " << i << " TT-Core [" << ri << ", " << shape[i] << ", " << r << "] NNZs: " << row_subset.nnz_count << std::endl;
            outFile << "Mode " << i << " Piv mat [" << ri << ", " << ri << "] NNZs: " << pivot_mat.nnz_count << std::endl; 
        } else {
            // Inverse merge kernel for reconstruction approximant
            auto Z = dcoeffZReconCPU(idResult.sparse_interp_coeff, idResult.cps, ri, col);

            // TT-cores
            if (i == dim - 1) {
                outFile << "TT representation (one-side ID format)" << std::endl;
                ResList.EndG.reset(Z.nnz_count, ri, shape[i]);
                ResList.EndG.nnz_count = Z.nnz_count;
                std::copy(Z.values, Z.values + Z.nnz_count, ResList.EndG.values);
                std::copy(Z.row_indices, Z.row_indices + Z.nnz_count, ResList.EndG.indices[0]);
                std::copy(Z.col_indices, Z.col_indices + Z.nnz_count, ResList.EndG.indices[1]);
            } else {
                ResList.InterG[i-1].reset(Z.nnz_count, ri, shape[i], r);
                ResList.InterG[i-1].nnz_count = Z.nnz_count;
                std::copy(Z.values, Z.values + Z.nnz_count, ResList.InterG[i-1].values);
                std::copy(Z.row_indices, Z.row_indices + Z.nnz_count, ResList.InterG[i-1].indices[0]);
                for (size_t n = 0; n < Z.nnz_count; ++n) {
                    ResList.InterG[i-1].indices[1][n] = Z.col_indices[n] / r;
                    ResList.InterG[i-1].indices[2][n] = Z.col_indices[n] % r;
                }
            }
            
            // Output NNZ info    
            outFile << "Mode " << i << " TT-Core [" << ri << ", " << shape[i] << ", " << r << "] NNZs: " << Z.nnz_count << std::endl;
        }

        // Form new W from the interpolative factor C
        nbar = nbar * ri;
        nbar = nbar / shape[i] / r; // New total size of W
        
        r = ri;
        W = W.subcol(idResult.pivot_cols, ri);

        idResult.freeSpInterpRes();
    }

    // Append the last factor
    ResList.StartG.reset(W.nnz_count, W.rows, W.cols);
    ResList.StartG.nnz_count = W.nnz_count;

    outFile << "Mode " << 0 << " Core [" << 1 << ", " << shape[0] << ", " << r << "] NNZs: " << W.nnz_count << std::endl;
    outFile.close();

    std::copy(W.values, W.values + W.nnz_count, ResList.StartG.values);
    std::copy(W.row_indices, W.row_indices + W.nnz_count, ResList.StartG.indices[0]);
    std::copy(W.col_indices, W.col_indices + W.nnz_count, ResList.StartG.indices[1]);
    return ResList;
}

// Explicitly instantiate the specializations
template SparseTTRes TT_ID_sparse<double, 3>(const COOTensor<double, 3>&, double, size_t, double, bool, bool);
template SparseTTRes TT_ID_sparse<double, 4>(const COOTensor<double, 4>&, double, size_t, double, bool, bool);
template SparseTTRes TT_ID_sparse<double, 5>(const COOTensor<double, 5>&, double, size_t, double, bool, bool);
template SparseTTRes TT_ID_sparse<double, 6>(const COOTensor<double, 6>&, double, size_t, double, bool, bool);