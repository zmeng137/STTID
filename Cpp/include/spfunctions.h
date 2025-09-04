#ifndef SPFUNCTIONS_H
#define SPFUNCTIONS_H

#include "structures.h"
#include "spmatrix.h"

// Sparse decompositions (CPU versions)
decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU_l1(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim, bool const isFullReturn);
decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU_l1(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim);
decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU_l2(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim, bool const isFullReturn);
decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU_l2(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim);
decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU_l3(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim, bool const isFullReturn);
decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU_l3(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim);

// Sparse decompositions (GPU versions)
decompRes::SparsePrrlduRes<double> dSparse_PartialRRLDU_GPU_l3(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, long long const maxdim, bool const isFullReturn);
decompRes::SparseInterpRes<double> dSparse_Interpolative_GPU_l3(COOMatrix_l2<double> const M, double const cutoff, double const spthres, long long const maxdim);

// Z reconstruction
COOMatrix_l2<double> dcoeffZReconCPU(double* coeffMatrix, long long* pivot_col, long long rank, long long col);
COOMatrix_l2<double> dcoeffZReconCPU(COOMatrix_l2<double> sparse_coeff_mat, std::unordered_map<long long, long long> cps, long long rank, long long col);

#endif