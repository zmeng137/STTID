#include <gtest/gtest.h>
#include "dtensor.h"
#include "dfunctions.h"
#include "densett.h"
#include "util.h"

TEST(PRRLDU_ID, DenseMat6by8)
{
    // Initialize the test matrix
    size_t Nr = 6, Nc = 8;
    double* M = new double[Nr * Nc] {
        1.0, 2.0, 3.0, 4.4231, 5.0, -8.3, 7.0, 0.2,
        9.0, 10.0, -11.0, 12.0, 13.23, 14.0, 15.0, 16.0,
        17.0, 18.232, 19.0, 20.0, 21.0, 22.432, 23.0, 24.0,
        25.3, 26.0, 20.345, 28.0, -9.1, 30.0, 31.0, 32.0,
        -33.211, 34.0, 3.5732, 36.0, 37.0, 38.0, 39.4323, 40.0,
        39.33, 42.0, 43.0, -41.21, 45.0, 46.0, 47.167, 48.0
    };

    // Interpolative decomposition based on partial rank revealing LDU decomposition
    float cutoff = 1e-8;
    size_t maxdim = 8, outdim;
    double* C = new double[Nr * maxdim]{0.0};
    double* Z = new double[maxdim * Nc]{0.0};
    dInterpolative_PrrLDU(M, Nr, Nc, maxdim, cutoff, C, Z, outdim);

    // Verification
    double max_error = 0.0;
    for (size_t i = 0; i < Nr; ++i)
        for (size_t j = 0; j < Nc; ++j) {
            double temp = 0.0;
            for (size_t k = 0; k < outdim; ++k)
                temp += C[i * outdim + k] * Z[k * Nc + j];
            max_error = std::max(max_error, std::abs(temp - M[i * Nc + j]));
        }

    EXPECT_NEAR(max_error, 0.0, 1e-10);

    delete[] M;
    delete[] C;
    delete[] Z;
}

TEST(PRRLDU_ID, Random)
{
// Initialize random rank-deficient matrix M 
    int Nr = 20, Nc = 15;
    int trueRank = 11;
    double* A = new double[Nr * trueRank];
    double* B = new double[trueRank * Nc];
    double* M = new double[Nr * Nc]{0.0};
    util::generateRandomArray(A, Nr * trueRank, -100.0, 100.0);
    util::generateRandomArray(B, trueRank * Nc, -100.0, 100.0);
    for (int i = 0; i < Nr; ++i) 
        for (int j = 0; j < Nc; ++j) 
            for (int k = 0; k < trueRank; ++k)
                M[i * Nc + j] += A[i * trueRank + k] * B[k * Nc + j];
    
    // Interpolative decomposition based on partial rank revealing LDU decomposition
    float cutoff = 1e-8;
    size_t maxdim = 15, outdim;
    double* C = new double[Nr * maxdim]{0.0};
    double* Z = new double[maxdim * Nc]{0.0};
    dInterpolative_PrrLDU(M, Nr, Nc, maxdim, cutoff, C, Z, outdim);

    // Verification
    double max_error = 0.0;
    for (size_t i = 0; i < Nr; ++i)
        for (size_t j = 0; j < Nc; ++j) {
            double temp = 0.0;
            for (size_t k = 0; k < outdim; ++k)
                temp += C[i * outdim + k] * Z[k * Nc + j];
            max_error = std::max(max_error, std::abs(temp - M[i * Nc + j]));
        }

    EXPECT_NEAR(max_error, 0.0, 1e-10);

    delete[] M;
    delete[] C;
    delete[] Z;
}

TEST(TTSVD_TEST, Way3_TTSVD_dense1)
{   
    tblis::tensor<double> T({2, 4, 3});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                T(i,j,k) = i + j + k;    

    auto factors = TT_SVD_dense(T, 2, 1E-5);
    auto tensor = denseT::TT_Contraction_dense(factors);    

    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k) 
                max_error = std::max(std::abs(T(i,j,k) - tensor(i,j,k)), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}

TEST(TTSVD_TEST, Way4_TTSVD_dense2)
{   
    tblis::tensor<double> T({5, 3, 5, 6});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                for (int l=0; l<5; ++l)
                   T(i,j,k,l) = i * j + k - l;    

    auto factors = TT_SVD_dense(T, 10, 1E-10);
    auto tensor = denseT::TT_Contraction_dense(factors);
        
    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                for (int l=0; l<5; ++l) 
                    max_error = std::max(std::abs(T(i,j,k,l) - tensor(i,j,k,l)), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}

TEST(TTID_PRRLDU_TEST, Way3_TTID_dense1)
{   
    tblis::tensor<double> T({2, 4, 3});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                T(i,j,k) = i + j + k;    

    auto factors = TT_IDPRRLDU_dense(T, 2, 1E-5);
    auto tensor = denseT::TT_Contraction_dense(factors);    

    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k) 
                max_error = std::max(std::abs(T(i,j,k) - tensor(i,j,k)), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}

TEST(TTID_PRRLDU_TEST, Way4_TTID_dense2)
{   
    tblis::tensor<double> T({5, 3, 5, 6});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                for (int l=0; l<5; ++l)
                   T(i,j,k,l) = i * j + k - l;    

    auto factors = TT_IDPRRLDU_dense(T, 10, 1E-10);
    auto tensor = denseT::TT_Contraction_dense(factors);
        
    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                for (int l=0; l<5; ++l) 
                    max_error = std::max(std::abs(T(i,j,k,l) - tensor(i,j,k,l)), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}