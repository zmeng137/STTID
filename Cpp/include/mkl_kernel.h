#ifndef MKL_KERNEL_H
#define MKL_KERNEL_H

#include "structures.h"

void mkl_trsv_idkernel(
        decompRes::SparseInterpRes<double> idResult, 
        decompRes::SparsePrrlduRes<double> prrlduResult,
        long long output_rank, long long Nc);

void denseLU_cpukernel(double* M_full, const long long Nr, const long long Nc, 
    long long& s, const long long k, const double cutoff, double& inf_error, 
    long long* rps, long long* cps, decompRes::SparsePrrlduRes<double> resultSet);

void denseTRSV_interp(decompRes::SparsePrrlduRes<double> prrlduResult, double* U11, double* b, 
    const long long Nc, const long long output_rank, decompRes::SparseInterpRes<double> idResult);

#endif