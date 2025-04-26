// structures.h - Data structures and result types
#ifndef STRUCTURES_H
#define STRUCTURES_H

#include "core.h"
#include "spmatrix.h"

namespace decompRes {

    template<class T>
    struct SparsePrrlduRes {
        // Dense factors
        T* d = nullptr;         // Diagonal entries
        T* dense_L = nullptr;   // Return data when fullReturn and not sparseReturn
        T* dense_U = nullptr;   // Return data when fullReturn and not sparseReturn 
        T* dense_U11 = nullptr; // Return data when not sparseReturn (square, U[:,:output_rank])
        T* dense_B = nullptr;   // Return data when not sparseReturn (coefficient right-hand side vectors)

        // Sparse factors
        COOMatrix_l2<T> sparse_L;   // Return data when fullReturn and sparseReturn
        COOMatrix_l2<T> sparse_U;   // Return data when fullReturn and sparseReturn
        COOMatrix_l2<T> sparse_U11; // Return data when sparseReturn (square, U[:,:output_rank])
        COOMatrix_l2<T> sparse_B;   // Return data when sparseReturn (coefficient right-hand side vectors)

        // Rank and permutation
        bool isSparseRes;   // If sparse result 
        bool isFullReturn;  // If full return
        T inf_error;   // Inference error
        long long rank;   // Real revealed rank
        long long output_rank; // Output rank
        long long* row_perm_inv = nullptr; // Inverse permutation row
        long long* col_perm_inv = nullptr; // Inverse permutation column
        long long* piv_cols = nullptr;     // Permutation column indices
        
        // Memory release
        void freeSpLduRes() {
            if (d != nullptr) delete[] d;
            if (dense_L != nullptr) delete[] dense_L;
            if (dense_U != nullptr) delete[] dense_U;
            if (dense_B != nullptr) delete[] dense_B;
            if (dense_U11 != nullptr) delete[] dense_U11;
            if (row_perm_inv != nullptr) delete[] row_perm_inv;
            if (col_perm_inv != nullptr) delete[] col_perm_inv;
            if (piv_cols != nullptr) delete[] piv_cols;
            //sparse_L.~COOMatrix_l2<T>();
            //sparse_U.~COOMatrix_l2<T>();
            //sparse_B.~COOMatrix_l2<T>();
            //sparse_U11.~COOMatrix_l2<T>();
        };
    };


    template<class T>
    struct SparseInterpRes {
        T* C = nullptr;  // Return data when fullReturn
        T* Z = nullptr;  // Return data when fullReturn
        T* interp_coeff = nullptr;    // Return data when not fullReturn
        long long* pivot_cols = nullptr; // Pivoted columns 
        long long rank;  // Detected rank of matrix
        long long output_rank; // Output(Truncated) rank
        bool isFullReturn;

        void freeSpInterpRes() {
            if (C != nullptr) delete[] C;
            if (Z != nullptr) delete[] Z;
            if (interp_coeff != nullptr) delete[] interp_coeff;
            if (pivot_cols != nullptr) delete[] pivot_cols;
        };
    };

}

enum class Distribution {
    UNIFORM,      // Uniform distribution between min and max
    NORMAL,       // Normal distribution with mean and standard deviation
    STANDARD_NORMAL,  // Normal distribution with mean=0, std=1
    GAMMA         // Gamma distribution with shape (k) and scale (theta)
};

struct DistributionParams {
    // Parameters for various distributions
    double min_value = -1.0;     // For uniform
    double max_value = 1.0;      // For uniform
    double mean = 0.0;           // For normal
    double std_dev = 1.0;        // For normal
    double gamma_shape = 1.0;    // k (shape) parameter for gamma
    double gamma_scale = 1.0;    // θ (scale) parameter for gamma
    
    // Constructor for uniform distribution
    static DistributionParams uniform(double min = -1.0, double max = 1.0) {
        DistributionParams params;
        params.min_value = min;
        params.max_value = max;
        return params;
    }
    
    // Constructor for normal distribution
    static DistributionParams normal(double mean = 0.0, double std_dev = 1.0) {
        DistributionParams params;
        params.mean = mean;
        params.std_dev = std_dev;
        return params;
    }
    
    // Constructor for gamma distribution
    static DistributionParams gamma(double shape = 1.0, double scale = 1.0) {
        DistributionParams params;
        params.gamma_shape = shape;
        params.gamma_scale = scale;
        return params;
    }
};

#endif // TENSOR_STRUCTURES_H