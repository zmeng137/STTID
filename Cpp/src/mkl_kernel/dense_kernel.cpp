#include "structures.h"
//#include "util.h"
#include <mkl/mkl.h>

void denseLU_cpukernel(double* M_full, const long long Nr, const long long Nc, 
    long long& s, const long long k, const double cutoff, double& inf_error, 
    long long* rps, long long* cps, decompRes::SparsePrrlduRes<double> resultSet)
{
    while (s < k) {
        // Partial M, Mabs = abs(M[s:,s:])    
        auto iter_start_time = std::chrono::high_resolution_clock::now();
        double Mabs_max = 0.0;
        long long piv_r;
        long long piv_c;
        for (long long i = s; i < Nr; ++i)
            for (long long j = s; j < Nc; ++j) {
                double Mabs = std::abs(M_full[i * Nc + j]);
                if (Mabs > Mabs_max) {
                    Mabs_max = Mabs;
                    piv_r = i;
                    piv_c = j;
                }   
            }
        
        // termination condition
        std::cout << "DenseLU: s=" << s << ", Mabs_max=" << Mabs_max << ", pivrc=" << piv_c << "," << piv_c;
        size_t memDense = sizeof(double) * Nr * Nc;
        std::cout << "-MEM(MB) of Dense M:" << memDense / 1024.0 / 1024.0;
        if (Mabs_max < cutoff) {
            inf_error = Mabs_max;
            break;
        }

        // Update diagonal entries
        resultSet.d[s] = M_full[piv_r * Nc + piv_c]; 

        // Row/Column swap
        cblas_dswap(Nc, M_full +  piv_r * Nc, 1, M_full + s * Nc, 1);
        cblas_dswap(Nr, M_full + piv_c, Nc, M_full + s, Nc);
        
        // Outer-product update 
        if (s < k - 1) {
            for (long long i = s + 1; i < Nr; ++i) {
                for (long long j = s + 1; j < Nc; ++j) {
                    double outprod = M_full[i * Nc + s] * M_full[s * Nc + j] / M_full[s * Nc + s];
                    M_full[i * Nc + j] = M_full[i * Nc + j] - outprod;
                }
            }
        }

        // Swap rps, cps
        long long temp;
        temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
        temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;

        s += 1;

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end_time - iter_start_time);
        double runtime_ms = duration.count();
        std::cout << ", rt(ms)=" << runtime_ms << std::endl;
    }
}

void denseTRSV_interp(decompRes::SparsePrrlduRes<double> prrlduResult, double* U11, double* b, 
    const long long Nc, const long long output_rank, decompRes::SparseInterpRes<double> idResult)
{
    // Extract relevant submatrices
    for (long long i = 0; i < output_rank; ++i)
        std::copy(prrlduResult.dense_U + i * Nc, prrlduResult.dense_U + i * Nc + output_rank, U11 + i * output_rank);

    // Compute the interpolative coefficients through solving upper triangular systems
    for (long long i = output_rank; i < Nc; ++i) {
        // Right hand side b (one column of the U)
        for (long long j = 0; j < output_rank; ++j)
            b[j] = prrlduResult.dense_U[j * Nc + i];

        // Triangular solver (BLAS)        
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, output_rank, U11, output_rank, b, 1);

        // Copy the solution to iU11 columns
        for (long long j = 0; j < output_rank; ++j) 
            idResult.interp_coeff[j * (Nc - output_rank) + (i - output_rank)] = b[j];                
    }
}