#include "spmatrix.h"
#include "structures.h"
#include "util.h"

// External MKL
#include <mkl/mkl.h>
#include <mkl/mkl_spblas.h>

void mkl_trsv_idkernel(
        decompRes::SparseInterpRes<double> idResult, 
        decompRes::SparsePrrlduRes<double> prrlduResult,
        long long output_rank, long long Nc)        
{
    // Sparse-COO U11 -> CSR U11
    long long U11_rows = prrlduResult.sparse_U11.rows;
    long long U11_cols = prrlduResult.sparse_U11.cols;
    long long U11_nnz = prrlduResult.sparse_U11.nnz_count;
    long long *U11_cooRows = prrlduResult.sparse_U11.row_indices;
    long long *U11_cooCols = prrlduResult.sparse_U11.col_indices;
    double *U11_cooVals = prrlduResult.sparse_U11.values;
    sparse_matrix_t cooU11, csrU11;
    util::CHECK_MKL_ERROR(mkl_sparse_d_create_coo(&cooU11, SPARSE_INDEX_BASE_ZERO,
                                                  U11_rows, U11_cols, U11_nnz, U11_cooRows, U11_cooCols, U11_cooVals));
    util::CHECK_MKL_ERROR(mkl_sparse_convert_csr(cooU11, SPARSE_OPERATION_NON_TRANSPOSE, &csrU11));

    // Sparse-COO B -> CSC B
    long long B_rows = prrlduResult.sparse_B.rows;
    long long B_cols = prrlduResult.sparse_B.cols;
    long long B_nnz = prrlduResult.sparse_B.nnz_count;
    long long *B_cooRows = prrlduResult.sparse_B.row_indices;
    long long *B_cooCols = prrlduResult.sparse_B.col_indices;
    double *B_cooVals = prrlduResult.sparse_B.values;
    sparse_matrix_t cooB, cscB;
    util::CHECK_MKL_ERROR(mkl_sparse_d_create_coo(&cooB, SPARSE_INDEX_BASE_ZERO,
                                                  B_rows, B_cols, B_nnz, B_cooRows, B_cooCols, B_cooVals));

    util::CHECK_MKL_ERROR(mkl_sparse_convert_csr(cooB, SPARSE_OPERATION_TRANSPOSE, &cscB));
    sparse_index_base_t indexing;
    long long *cscB_col_start;
    long long *cscB_col_end;
    long long *cscB_row_ind;
    double *cscB_values;
    util::CHECK_MKL_ERROR(mkl_sparse_d_export_csr(cscB, &indexing, &B_rows, &B_cols,
                                                  &cscB_col_start, &cscB_col_end, &cscB_row_ind, &cscB_values));

    double *b = new double[output_rank]{0.0};
    double *x = new double[output_rank]{0.0};

    // Create matrix descriptor
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    // Compute the interpolative coefficients through solving upper triangular systems
    std::cout << "Sptrsv (MKL sparse blase CPU) for coefficients starts.\n";
    for (long long i = 0; i < Nc - output_rank; ++i)
    {
        // Right hand side b (one column of the U)
        memset(b, 0, sizeof(double) * output_rank);
        long long row_end = cscB_col_start[i + 1];
        long long row_start = cscB_col_start[i];
        long long interval = row_end - row_start;
        for (long long j = 0; j < interval; ++j)
        {
            double tval = cscB_values[row_start + j];
            long long bri = cscB_row_ind[row_start + j];
            b[bri] = tval;
        }

        // MKL sparse triangular solver
        {util::Timer timer("mklSpSV solves");
        util::CHECK_MKL_ERROR(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrU11, descr, b, x));}

        // Copy the solution to iU11 columns
        //for (long long j = 0; j < output_rank; ++j)
        //    idResult.interp_coeff[j * (Nc - output_rank) + i] = x[j];
    }

    // Clean up
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cooU11));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(csrU11));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cooB));
    util::CHECK_MKL_ERROR(mkl_sparse_destroy(cscB));
    delete[] b;
    delete[] x;
    std::cout << "Sptrsv (MKL sparse blase CPU) for coefficients ends.\n";
}
        
        
