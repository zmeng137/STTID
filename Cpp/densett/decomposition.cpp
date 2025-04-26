#include "dfunctions.h"
#include "util.h"
#include <cblas.h>
#include <lapacke.h>

void dQR_MGS(double* M, int Nr, int Nc, double* Q, double* R) {
    for (int j = 0; j < Nc; j++) {
        // Compute the j-th column of Q
        for (int i = 0; i < Nr; i++) {
            Q[i * Nc + j] = M[i * Nc + j];
        }

        for (int i = 0; i < j; i++) {
            double dot_product = 0.0;
            for (int k = 0; k < Nr; k++) {
                dot_product += Q[k * Nc + i] * Q[k * Nc + j];
            }
            R[i * Nc + j] = dot_product;
            for (int k = 0; k < Nr; k++) {
                Q[k * Nc + j] -= R[i * Nc + j] * Q[k * Nc + i];
            }
        }

        double norm = 0.0;
        for (int k = 0; k < Nr; k++) {
            norm += Q[k * Nc + j] * Q[k * Nc + j];
        }
        norm = std::sqrt(norm);
        R[j * Nc + j] = norm;
        for (int k = 0; k < Nr; k++) {
            Q[k * Nc + j] /= norm;
        }
    }
    return;
}

void dPivotedQR_MGS(double* A, int Nr, int Nc, double* Q, double* R, int* P, int& rank)
{   
    // Copy the input matrix
    double* M = new double[Nr * Nc];
    std::copy(A, A + Nr * Nc, M);

    // v_j = ||X[:,j]||^2, j=1,...,n
    double* v = new double[Nc]{0.0};
    blas_dcolumn_inner_products(M, Nr, Nc, v);

    // Determine an index p1 such that v_p1 is maximal
    double* max_ptr_v = std::max_element(v, v + Nc);
    int pk = std::distance(v, max_ptr_v);  

    // Initialization of arrays
    std::iota(P, P + Nc, 0);        // Fill the permutation array with 0, 1, 2, ..., Nc.
    std::fill(Q, Q + Nc * Nr, 0.0); // Fill Q with zeros
    std::fill(R, R + Nc * Nc, 0.0); // Fill R with zeros

    // Modified Gram-Schmidt Process (To be modified? MGS)
    rank = 0;
    for (int k = 0; k < Nc; ++k) {
        // Swap arrays: X, v, P, R 
        cblas_dswap(Nr, M + pk, Nc, M + k, Nc); // Swap the pk-th and j-th column of M (To be optimized?)
        cblas_dswap(1, v + pk, 1, v + k, 1);    // Swap v[k] <-> v[pk]
        cblas_dswap(k, R + pk, Nc, R + k, Nc);  // Swap R[0:k,pk] <-> R[0:k,k]  
        int temp = P[k];    // Swap P[k] <-> P[pk]
        P[k] = P[pk];
        P[pk] = temp;

        // I can use blas but I write my own code here for future optimization
        for (int i = 0; i < Nr; ++i) {
            double temp = 0.0;
            for (int j = 0; j < k; ++j) 
                temp += Q[i * Nc + j] * R[j * Nc + k];
            Q[i * Nc + k] = M[i * Nc + k] - temp;
        }

        double inner_prod = 0.0;
        for (int i = 0; i < Nr; ++i) 
            inner_prod += Q[i * Nc + k] * Q[i * Nc + k];
        R[k * Nc + k] = std::sqrt(inner_prod);

        for (int i = 0; i < Nr; ++i) 
            Q[i * Nc + k] = Q[i * Nc + k] / R[k * Nc + k];

        for (int i = k + 1; i < Nc; ++i) {
            double temp = 0.0;
            for (int j = 0; j < Nr; ++j) 
                temp += Q[j * Nc + k] * M[j * Nc + i];
            R[k * Nc + i] = temp;
        }
        
        // Rank increment
        rank += 1;    
        // Update v_j
        for (int j = k + 1; j < Nc; ++j) 
            v[j] = v[j] - R[k * Nc + j] * R[k * Nc + j];

        // Determine an index p1 such that v_p1 is maximal
        max_ptr_v = std::max_element(v + k + 1, v + Nc);
        pk = std::distance(v, max_ptr_v);  

        // Rank revealing step
        // PROBLEM! We need to find how to determine the rank cutoff tolerance!
        // Sometimes 1
        if (v[pk] < 1) 
            break;
    }

    delete[] v;
    delete[] M;
    return;
}

decompRes::PrrlduRes<double> 
dPartialRRLDU(double* M_, size_t Nr, size_t Nc,
              double cutoff, size_t maxdim, size_t mindim) 
{
    // Dimension argument check
    util::Timer timer("dPartialRRLDU");
    assertm(maxdim > 0, "maxdim must be positive");
    assertm(mindim > 0, "mindim must be positive");
    mindim = std::min(maxdim, mindim);

    // Copy input M_ to a M (to be optimized?)
    double* M = new double[Nr * Nc];
    std::copy(M_, M_ + Nr * Nc, M);
    size_t k = std::min(Nr, Nc);
    
    // Initialize permutations
    size_t* rps = new size_t[Nr];
    size_t* cps = new size_t[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);

    // Find pivots
    double inf_error = 0.0;
    size_t s = 0;
    while (s < k) {
        std::cout << "PRRLU Iter " << s;
        // Partial M, Mabs = abs(M[s:,s:])
        size_t subN = (Nr - s) * (Nc - s);
        double* Mabs = new double[subN];
        for (size_t i = 0; i < Nr - s; ++i)
            for (size_t j = 0; j < Nc - s; ++j)
                Mabs[i * (Nc - s) + j] = std::abs(M[(i + s) * Nc + (j + s)]);

        // Max value of Mabs
        double* pMabs_max = std::max_element(Mabs, Mabs + subN);
        double Mabs_max = *pMabs_max;
        if (Mabs_max < cutoff) {
            inf_error = Mabs_max;
            delete[] Mabs;
            break;
        }

        // piv, swap rows and columns
        size_t max_idx = std::distance(Mabs, pMabs_max);
        size_t piv_r = max_idx / (Nc - s) + s;
        size_t piv_c = max_idx % (Nc - s) + s;
        cblas_dswap(Nc, M +  piv_r * Nc, 1, M + s * Nc, 1);
        cblas_dswap(Nr, M + piv_c, Nc, M + s, Nc);

        if (s < k - 1) {
            for (size_t i = s + 1; i < Nr; ++i) {
                for (size_t j = s + 1; j < Nc; ++j) {
                    double outprod = M[i * Nc + s] * M[s * Nc + j] / M[s * Nc + s];
                    M[i * Nc + j] = M[i * Nc + j] - outprod;
                }
            }
        }

        // Swap rps, cps
        size_t temp;
        temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
        temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;

        delete[] Mabs;
        s += 1;

        std::ifstream status("/proc/self/status");
        std::string line;
        long virtualMem = 0;
        long residentMem = 0;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmSize") {
                virtualMem = std::stol(line.substr(line.find_first_of("0123456789")));
            }
            if (line.substr(0, 6) == "VmRSS") {
                residentMem = std::stol(line.substr(line.find_first_of("0123456789")));
                break;  // Found both values we need
            }
        }
        std::cout << "-MEM(MB):Virtual:" << virtualMem/1024.0
                << ",Resident:" << residentMem/1024.0 << std::endl;
    } 

    // Commented on Jan 7, 2025
    // New pivoted matrix M
    //for (size_t i = 0; i < Nr; ++i)
    //    for (size_t j = 0; j < Nc; ++j)
    //        M[i * Nc + j] = M_[rps[i] * Nc + cps[j]];
    //std::cout << "M\n";
    //util::PrintMatWindow(M, Nr, Nc, {0,Nr-1}, {0,Nc-1});
    
    // Initialize LDU
    double* L = new double[Nr * k]{0.0};
    double* d = new double[k]{0.0};
    double* U = new double[k * Nc]{0.0};
    size_t rank = 0;
    for (size_t i = 0; i < std::min(k, Nr); ++i)
        L[i * k + i] = 1.0;
    for (size_t i = 0; i < std::min(k, Nc); ++i)
        U[i * Nc + i] = 1.0;

    
    // Rank-revealing Guassian elimination
    for (size_t s = 0; s < std::min(k, maxdim); ++s) {
        double P = M[s * Nc + s];
        d[s] = P; 

        // Termination condition
        if (rank < mindim) {
            // Do nothing
        } else if (P == 0 || (std::abs(P) < cutoff && rank + 1 > mindim)) {
            break;
        }

        // Commented on Jan 8, 2025
        //if (P == 0) 
        //    P = 1;
        
        rank += 1;

        // Gaussian elimination
        if (s < Nr - 1) {
            // pivoted col
            for (size_t i = s + 1; i < Nr; ++i)
                L[i * k + s] = M[i * Nc + s] / P;
        }
        if (s < Nc - 1) {
            // pivoted row
            for (size_t j = s + 1; j < Nc; ++j)
                U[s * Nc + j] = M[s * Nc + j] / P;
        }
        // Commented on Jan 7, 2025
        //if (s < k - 1) {
        //    for (size_t i = s + 1; i < Nr; ++i)
        //        for (size_t j = s + 1; j < Nc; ++j)
        //            M[i * Nc + j] = M[i * Nc + j] - M[i * Nc + s] * M[s * Nc + j] / P;
        //}
    }

    // Initialize the returned result struct
    decompRes::PrrlduRes<double> resultSet;
    resultSet.L = new double[Nr * rank];
    resultSet.d = new double[rank];
    resultSet.U = new double[rank * Nc];
    resultSet.rank = rank;
    resultSet.col_perm_inv = new size_t[Nc]{0};
    resultSet.row_perm_inv = new size_t[Nr]{0};
    resultSet.inf_error = inf_error;
    
    // Copy results
    std::copy(U, U + rank * Nc, resultSet.U);
    std::copy(d, d + rank, resultSet.d);
    for (size_t i = 0; i < Nr; ++i)
        std::copy(L + i * k, L + i * k + rank, resultSet.L + i * rank);
    // Create inverse permutations
    for (size_t i = 0; i < Nr; ++i)
        resultSet.row_perm_inv[rps[i]] = i;
    for (size_t j = 0; j < Nc; ++j)
        resultSet.col_perm_inv[cps[j]] = j;
    
    // Memory release
    delete[] M;
    delete[] rps;
    delete[] cps;
    delete[] L;
    delete[] d;
    delete[] U;

    return resultSet;
}