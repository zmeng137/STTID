#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <cstdint>

// Format manipulation cusparse
void cusparse_dcooRowSort(int num_rows, int num_columns, int nnz, int* d_rows, int* d_columns, double* d_values);
void cusparse_dcooColSort(int num_rows, int num_columns, int nnz, int* d_rows, int* d_columns, double* d_values);
void cusparse_dcoo2csr(int num_rows, int num_columns, int nnz, int* d_cooRows, int* d_cooCols, double* d_cooVals, int* d_csrRows);
void cusparse_dcoo2csc(int num_rows, int num_columns, int nnz, int* d_cooRows, int* d_cooCols, double* d_cooVals, int* d_cscCols);
void cud_coo2cs(const int64_t num_rows, const int64_t num_columns, int64_t nnz, int64_t* d_cooRows, int64_t* d_cooCols, double* d_cooVals, int64_t* d_csPtrs, bool row_order = true);
void cusparse_uptrsv_csr_64(int64_t rows, int64_t cols, int64_t nnz, int64_t* d_csrRowPtr, int64_t* d_csrColInd, double* d_csrVal, double* d_b, double* d_x, double alpha);
void denseb_assign_gpu(double* d_b, int64_t pos, int64_t* d_B_cscCols, int64_t* d_B_cooRows, double* d_B_cooVals);

// classification function on gpu
void eleClassify_gpu(
    const long long* const Ms_d_row_indices, const long long* const Ms_d_col_indices, const double* const Ms_d_values, long long const Ms_nnz,  // Read-only Ms on device 
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long& Mp_nnz,                                           // To-be-written Mp on device 
    long long* Mt_hash_d_key, double* Mt_hash_d_val, long long& Mt_hash_nnz, long long const Mt_col_num,                                        // To-be-written Mt on device
    long long* vr_d_idx, double* vr_d_val, long long& vr_nnz, long long* vc_d_idx, double* vc_d_val, long long& vc_nnz,                         // To-be-written Vr/Vc on device
    long long const s);

// outer product Gaussian elimination on gpu
unsigned long long outerproduct_update_gpu(
    long long*& Mt_hash_d_key, double*& Mt_hash_d_val, long long& Mt_hash_nnz,               // In/Out: Hash table Mt
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,   // In: Sparse Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,   // In: Sparse Vc
    double const Mdenom, long long const M_col_num);                                         // In: Denominator and matrix column size

// outer product Gaussian elimination on gpu
unsigned long long outerproduct_update_gpu_opt(
    long long*& Mt_hash_d_key, double*& Mt_hash_d_val, long long& Mt_hash_nnz,               // In/Out: Hash table Mt
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,   // In: Sparse Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,   // In: Sparse Vc
    double const Mdenom, long long const M_col_num);                                         // In: Denominator and matrix column size

// Hash table -> COO format array
void hash2COO_gpu(long long* Ms_d_row_indices, long long* Ms_d_col_indices, double* Ms_d_values, long long& Ms_nnz,                   // Out: To-be-updated Ms
    const long long* const Mt_hash_d_key, const double* const Mt_hash_d_val, long long const Mt_hash_nnz, long long const col_num);    // In: Mt hash key-val pairs

// For pivoting
void device_warmup();
void findMaxAbsValueCublas(const double* const d_input, long long const n, long long& max_idx, double& max_val);
void findMaxAbsValueCublas(const double* d_input, int64_t n, int64_t& max_idx, double& max_val);
void coo_pivot_gpu(long long* d_row_indices, long long* d_col_indices, long long nnz_count, long long piv_r, long long piv_c, long long s);
void perm_inv_gpu(long long* d_pivot_cols, long long* d_col_perm_inv, long long Nc);

// Fusion kernel
unsigned long long A2345_fusion( 
    long long*& Ms_d_row_indices, long long*& Ms_d_col_indices, double*& Ms_d_values, long long& Ms_nnz, long long& Ms_capacity,   
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long& Mp_nnz,                            
    long long* Mt_hash_d_key, double* Mt_hash_d_val, long long& Mt_hash_nnz,                           
    long long* vr_d_idx, double* vr_d_val, long long& vr_nnz, 
    long long* vc_d_idx, double* vc_d_val, long long& vc_nnz,  
    long long const piv_r, long long const piv_c, long long const M_col_num, double const Mdenom, long long const s);

#endif