// functions.h - Matrix decomposition / Tensor train functions
#ifndef DFUNCTIONS_H
#define DFUNCTIONS_H

#include "core.h"

namespace decompRes
{
    template <class T>
    struct PrrlduRes
    {
        T *L = nullptr;
        T *d = nullptr;
        T *U = nullptr;
        size_t rank;
        size_t *row_perm_inv = nullptr;
        size_t *col_perm_inv = nullptr;
        T inf_error;

        // Memory release
        void freeLduRes()
        {
            if (L != nullptr)
                delete[] L;
            if (d != nullptr)
                delete[] d;
            if (U != nullptr)
                delete[] U;
            if (row_perm_inv != nullptr)
                delete[] row_perm_inv;
            if (col_perm_inv != nullptr)
                delete[] col_perm_inv;
        };
    };
}

// BLAS operations
void blas_dcolumn_inner_products(const double* A, int m, int n, double* results);

// SVD related functions
void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);
                                                   
// QR related functions                                                                                                                         
void dQR_MGS(double* M, int Nr, int Nc, double* Q, double* R);                                                             
double verifyQR(int m, int n, double* Q, double* R, double* A, int* jpvt);
void dPivotedQR(int m, int n, double* A, double* Q, double* R, int* jpvt);
void dPivotedQR_MGS(double* M, int Nr, int Nc, double* Q, double* R, int* P, int& rank);

// Partial rank-revealing LDU decomposition
decompRes::PrrlduRes<double> 
dPartialRRLDU(double* M_, size_t Nr, size_t Nc, double cutoff, size_t maxdim, size_t mindim);

// Interpolative decomposition
void dInterpolative_PivotedQR(double* A, int m, int n, int maxdim, double* C, double* Z, int& outdim);
void dInterpolative_PrrLDU(double* M, size_t Nr, size_t Nc, size_t maxdim, double cutoff, double* C, double* Z, size_t& outdim);

#endif 