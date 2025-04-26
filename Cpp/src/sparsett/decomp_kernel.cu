#include "util.h"
#include "cutil.h"

// cuCollection for hash map
#include <cuco/static_map.cuh>

// CUDA Thrust
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cuda/functional>

// Standard
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

//#define SHMEM_OPT

// To enbale double/float-type value in cuCollection map
namespace cuco {
    template <>
    struct is_bitwise_comparable<double> : std::true_type {};
}

// Block size and grid size (to be changed)
auto constexpr block_size= 512;
auto constexpr grid_size = 512;

// Define a sparse threshold
constexpr double SPVAL_EPS = 1e-14;

// Print tools: hash key array
__global__ void printKeyArray(long long* d_arr, long long size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < size) {
        printf("key_arr[%lld] = %lld \n", idx, d_arr[idx]);  // Note: %lld for long long int type 
        idx += gridDim.x * blockDim.x;
    }
}

// Print tools: hash value array
__global__ void printDoubleArray(double* d_arr, long long size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < size) {
        printf("d_arr[%lld] = %f\n", idx, d_arr[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

// Print tools: hash key-value pair array
__global__ void printKVArray(long long* d_Key, double* d_Val, long long size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < size) {
        printf("Key[%lld] = %lld, Val[%lld] = %f\n", idx, d_Key[idx], idx, d_Val[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

// Used to warm up the device
__global__ void warmup_kernel() 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = 1.0f;
    float b = 2.0f;

    for (int i = 0; i < 100; i++) {
        a = a * b + tid;
        b = a / (b + 1.0f);
    }
    
    if (tid < 0) {
        printf("This will never be printed: %f\n", b);
    }
}

// Get pivot reverse
__global__ void perm_inv_kernel(long long* pivot_cols, long long* col_perm_inv, long long Nc)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < Nc) {
        long long idx;
        for (long long j = 0; j < Nc; ++j) {
            if (col_perm_inv[j] == tid) {
                idx = j;
                break;
            }     
        }
        pivot_cols[tid] = idx;
        tid += gridDim.x * blockDim.x;
    }
}

// Value increment kernel for cuco static hashmap
template <typename Map, typename KeyIter, typename ValIter>
__global__ void increment_values(Map map_ref, KeyIter key_begin, ValIter val_begin, long long num_keys, 
                                KeyIter insertKey, ValIter insertVal, unsigned long long* num_inserted)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
  
    while (tid < num_keys) {
        // If the key exists in the map, find returns an iterator to the specified key. Otherwise it returns map.end()
        auto found = map_ref.find(key_begin[tid]);
        if (found != map_ref.end()) {
            // If the key exists, atomically increment the associated value
            auto ref = cuda::atomic_ref<typename Map::mapped_type, cuda::thread_scope_device>{found->second};
            ref.fetch_add(val_begin[tid], cuda::memory_order_relaxed);
        } else {
            unsigned long long old_idx = atomicAdd(num_inserted, 1);
            insertKey[old_idx] = key_begin[tid];
            insertVal[old_idx] = val_begin[tid];
        }
        
        tid += gridDim.x * blockDim.x;
    }
}

// Insertion kernel for cuco static hashmap
template <typename Map, typename KeyIter, typename ValIter>
__global__ void insert_kernel(Map map_ref, KeyIter key_begin, ValIter value_begin, long long num_keys)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_keys) {  
        map_ref.insert(cuco::pair{key_begin[tid], value_begin[tid]});
        tid += gridDim.x * blockDim.x;
    }
}

// Pivot COO-format matrix row/column kernel
__global__ void coo_pivot_kernel(long long* d_row_indices, long long* d_col_indices, long long nnz_count, 
    long long piv_r, long long piv_c, long long s)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < nnz_count) {
        if (d_row_indices[tid] == s)
            d_row_indices[tid] = piv_r;
        else if (d_row_indices[tid] == piv_r) 
            d_row_indices[tid] = s;
        if (d_col_indices[tid] == s)
            d_col_indices[tid] = piv_c;
        else if (d_col_indices[tid] == piv_c) 
            d_col_indices[tid] = s;            
        tid += blockDim.x * gridDim.x;
    }
}

// Return a key of a hash table
__device__ long long cal_hashKey(long long row, long long col, long long col_num) 
{
    long long key = row * col_num + col;
    if (key < 0) printf("NOTE! KEY BUG!\n");
    return key;
}

// Lookup cuco Hash map
template <typename findMap>
__global__ void cuco_lookup_kernel(findMap find_ref, long long const col_num,        
    const long long* const vr_d_idx, long long const vr_nnz,
    const long long* const vc_d_idx, long long const vc_nnz)
{
    long long xid = threadIdx.x + blockIdx.x * blockDim.x;
    long long yid = threadIdx.y + blockIdx.y * blockDim.y;
    
    while (xid < vr_nnz && yid < vc_nnz) {
        long long hashKey = cal_hashKey(vc_d_idx[yid], vr_d_idx[xid], col_num);
        auto found = find_ref.find(hashKey);
        if (found != find_ref.end()) {
            printf("vc,vr=%lld,%lld, Mt hash val=%f\n", vc_d_idx[yid], vr_d_idx[xid], found->second);       
        } 
        
        xid += blockDim.x * gridDim.x;
        yid += blockDim.y * gridDim.y;
    }
}

// Outer-product of Vector r and Vector c
// -----> x/vr
// | outer
// |     product  ==> HASH TABLE +=
// ↓ y/vc
template <typename Map, typename KeyIter, typename ValIter>
__global__ void outerproduct_update_kernel(Map map_ref, KeyIter* insertKey, ValIter* insertVal, // cuco hash map references for find/insert
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,      // Sparse vector Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,      // Sparse vector Vc
    double const Mdenom, long long const col_num, unsigned long long* num_inserted)             // Essential constants
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < (vr_nnz * vc_nnz)) {
        // Indices of vc/vr
        long long vc_idx = tid / vr_nnz;
        long long vr_idx = tid % vr_nnz;
        
        // Get the hash key-value pair used to update hash table
        KeyIter upd_hash_key = cal_hashKey(vc_d_idx[vc_idx], vr_d_idx[vr_idx], col_num); 
        ValIter upd_hash_val = -1.0 * vc_d_val[vc_idx] * vr_d_val[vr_idx] / Mdenom;

        // Search the hash map
        auto found = map_ref.find(upd_hash_key);
        if (found != map_ref.end()) {
            // If the key exists, atomically increment the associated value
            auto ref = cuda::atomic_ref<typename Map::mapped_type, cuda::thread_scope_device>{found->second};
            ref.fetch_add(upd_hash_val, cuda::memory_order_relaxed);
        } else {
            // If the key does not exist, append/insert the new key-val pair directly to the device array for key-val pairs
            unsigned long long old_idx = atomicAdd(num_inserted, 1);
            insertKey[old_idx] = upd_hash_key;
            insertVal[old_idx] = upd_hash_val;
        }

        // Next grid
        tid += blockDim.x * gridDim.x;
    }
}

// Trying to use shared memory in outerproduct_update kernel (No use)
template <typename Map, typename KeyIter, typename ValIter>
__global__ void outerproduct_update_kernel_sm(Map map_ref, KeyIter* insertKey, ValIter* insertVal, // cuco hash map references for find/insert
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,         // Sparse vector Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,         // Sparse vector Vc
    double const Mdenom, long long const col_num, unsigned long long* num_inserted)                // Essential constants
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long tx  = threadIdx.x; 

    __shared__ long long s_vr_idx[block_size];
    __shared__ long long s_vc_idx[block_size];
    __shared__ double s_vr_val[block_size];
    __shared__ double s_vc_val[block_size];

    while (tid < (vr_nnz * vc_nnz)) {
        // Indices of vc/vr
        long long vc_idx = tid / vr_nnz;
        long long vr_idx = tid % vr_nnz;

        s_vr_idx[tx] = vr_d_idx[vr_idx];
        s_vc_idx[tx] = vc_d_idx[vc_idx];
        s_vr_val[tx] = vr_d_val[vr_idx];
        s_vc_val[tx] = vc_d_val[vc_idx];
        __syncthreads();
        
        // Get the hash key-value pair used to update hash table
        KeyIter upd_hash_key = cal_hashKey(s_vc_idx[tx], s_vr_idx[tx], col_num); 
        ValIter upd_hash_val = -1.0 * s_vc_val[tx] * s_vr_val[tx] / Mdenom;

        // Search the hash map
        auto found = map_ref.find(upd_hash_key);
        if (found != map_ref.end()) {
            // If the key exists, atomically increment the associated value
            auto ref = cuda::atomic_ref<typename Map::mapped_type, cuda::thread_scope_device>{found->second};
            ref.fetch_add(upd_hash_val, cuda::memory_order_relaxed);
        } else {
            // If the key does not exist, append/insert the new key-val pair directly to the device array for key-val pairs
            unsigned long long old_idx = atomicAdd(num_inserted, 1);
            insertKey[old_idx] = upd_hash_key;
            insertVal[old_idx] = upd_hash_val;
        }

        // Next grid
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void outer_product_kernel(
    long long* d_hash_key, double* d_hash_val,                                              // Output key-val pairs
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,  // Sparse vector Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,  // Sparse vector Vc
    double const Mdenom, long long const col_num)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < (vr_nnz * vc_nnz)) {
        long long vc_idx = idx / vr_nnz;
        long long vr_idx = idx % vr_nnz;

        d_hash_key[idx] = cal_hashKey(vc_d_idx[vc_idx], vr_d_idx[vr_idx], col_num); 
        d_hash_val[idx] = -1.0 * vc_d_val[vc_idx] * vr_d_val[vr_idx] / Mdenom;

        idx += blockDim.x * gridDim.x;
    }
}

// Classify elements of Ms
__global__ void eleClassify_kernel(    
    const long long* const Ms_d_row_indices, const long long* const Ms_d_col_indices, const double* const Ms_d_values, long long const Ms_nnz,  // Read-only Ms on device 
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, unsigned long long* d_Mp_nnz,                                // To-be-written Mp on device
    long long* Mt_hash_d_key, double* Mt_hash_d_val, unsigned long long* d_Mt_hash_nnz, long long const Mt_col_num,                             // To-be-written Mt hash table on device
    long long* vr_d_idx, double* vr_d_val, unsigned long long* d_vr_nnz, long long* vc_d_idx, double* vc_d_val, unsigned long long* d_vc_nnz,   // To-be-written Vr and Vc on device
    long long const s)
{
    long long tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    while (tid < Ms_nnz) {
        long long r_i = Ms_d_row_indices[tid];
        long long c_i = Ms_d_col_indices[tid];
        double  v_i = Ms_d_values[tid];
    
        if (fabs(v_i) > SPVAL_EPS) {
            if (r_i > s && c_i > s) {
                // Add the element to hash array
                unsigned long long old_Mt_hash_nnz = atomicAdd(d_Mt_hash_nnz, 1);           
                long long hashKey = cal_hashKey(r_i, c_i, Mt_col_num);
                Mt_hash_d_key[old_Mt_hash_nnz] = hashKey;
                Mt_hash_d_val[old_Mt_hash_nnz] = v_i;           
            } else {
                // Add the element to Mp
                unsigned long long old_Mp_nnz = atomicAdd(d_Mp_nnz, 1);
                Mp_d_row_indices[old_Mp_nnz] = r_i;
                Mp_d_col_indices[old_Mp_nnz] = c_i;
                Mp_d_values[old_Mp_nnz] = v_i;
                if (r_i != c_i) {
                    if (r_i == s) {
                        // Add the element to Vr
                        unsigned long long old_vr_nnz = atomicAdd(d_vr_nnz, 1);
                        vr_d_idx[old_vr_nnz] = c_i;
                        vr_d_val[old_vr_nnz] = v_i;
                    }
                    if (c_i == s) {
                        // Add the element to Vc
                        unsigned long long old_vc_nnz = atomicAdd(d_vc_nnz, 1);
                        vc_d_idx[old_vc_nnz] = r_i;
                        vc_d_val[old_vc_nnz] = v_i;
                    }
                }   
            }
        }
        
        tid += blockDim.x * gridDim.x;
    }
}

// Convert hash table to COO format
__global__ void hash2COO_kernel(
    long long* Ms_d_row_indices, long long* Ms_d_col_indices, double* Ms_d_values, long long const col_num,
    const long long* const Mt_hash_d_key, const double* const Mt_hash_d_val, long long const Mt_hash_nnz)
{
    long long tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < Mt_hash_nnz) {
        // Access key-val pairs
        long long hKey = Mt_hash_d_key[tid];
        double hVal = Mt_hash_d_val[tid];
        
        // Get coordinates
        Ms_d_row_indices[tid] = hKey / col_num;
        Ms_d_col_indices[tid] = hKey % col_num;
        Ms_d_values[tid] = hVal;

        tid += blockDim.x * gridDim.x;
    }
}

// Function for cuda device warm-up
void device_warmup()
{ 
    // Run the kernel a few times
    for (int i = 0; i < 10; i++) {
        warmup_kernel<<<32, 256>>>();
        cudaDeviceSynchronize();
    }
    util::CHECK_LAST_CUDART_ERROR();
}

// Find maximum absolute value of on-device array
void findMaxAbsValueCublas(const double* const d_input, long long const n, long long& max_idx, double& max_val) 
{
    util::Timer timer("PRRLDU (GPU) - Sparse phase A1");
    cublasHandle_t handle;
    util::CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    int64_t idx;
    double result;
    if (n > 0) {
        util::CHECK_CUBLAS_ERROR(cublasIdamax_64(handle, n, d_input, 1, &idx));  // idx-64-bit cublas absolute max value
        util::CHECK_CUDART_ERROR(cudaMemcpy(&result, d_input + idx - 1, sizeof(double), cudaMemcpyDeviceToHost));
        max_val = result;
        max_idx = idx - 1;
    } else {
        max_val = 0;
        max_idx = -1;
        return;
    }
    
    //util::CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    util::CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}

void findMaxAbsValueCublas(const double* d_input, int64_t n, int64_t& max_idx, double& max_val)
{
    util::Timer timer("PRRLDU (GPU) - Sparse phase A1");
    cublasHandle_t handle;
    util::CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    int64_t idx;
    double result;
    if (n > 0) {
        util::CHECK_CUBLAS_ERROR(cublasIdamax_64(handle, n, d_input, 1, &idx));  // idx-64-bit cublas absolute max value
        util::CHECK_CUDART_ERROR(cudaMemcpy(&result, d_input + idx - 1, sizeof(double), cudaMemcpyDeviceToHost));
        max_val = result;
        max_idx = idx - 1;
    } else {
        max_val = 0;
        max_idx = -1;
        return;
    }
    
    //util::CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    util::CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}

void coo_pivot_gpu(long long* d_row_indices, long long* d_col_indices, long long nnz_count, 
    long long piv_r, long long piv_c, long long s)
{
    //size_t shared_mem_size = 2 * block_size * sizeof(long long);
    //coo_pivot_kernel_shared<<<grid_size, block_size, shared_mem_size>>>(d_row_indices, d_col_indices, nnz_count, piv_r, piv_c, s);
    //coo_pivot_kernel_improved<<<grid_size, block_size, shared_mem_size>>>(d_row_indices, d_col_indices, nnz_count, piv_r, piv_c, s);
    util::Timer timer("PRRLDU (GPU) - Sparse phase A2");
    coo_pivot_kernel<<<grid_size, block_size>>>(d_row_indices, d_col_indices, nnz_count, piv_r, piv_c, s);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();
}

// Data load: Classify and load elements to Mp, Mt, vr, and vc
void eleClassify_gpu(
    const long long* const Ms_d_row_indices, const long long* const Ms_d_col_indices, const double* const Ms_d_values, long long const Ms_nnz,  // Read-only Ms on device 
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long& Mp_nnz,                                           // To-be-written Mp on device 
    long long* Mt_hash_d_key, double* Mt_hash_d_val, long long& Mt_hash_nnz, long long const Mt_col_num,                                        // To-be-written Mt on device
    long long* vr_d_idx, double* vr_d_val, long long& vr_nnz, long long* vc_d_idx, double* vc_d_val, long long& vc_nnz,                         // To-be-written Vr/Vc on device
    long long const s)
{  
    util::Timer timer("PRRLDU (GPU) - Sparse phase A3");
    // nnz memory management
    unsigned long long h_Mp_nnz = Mp_nnz, h_Mt_hash_nnz = 0, h_vr_nnz = 0, h_vc_nnz = 0;
    unsigned long long* d_Mp_nnz, *d_Mt_hash_nnz, *d_vr_nnz, *d_vc_nnz;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_Mp_nnz,      sizeof(unsigned long long))); 
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_Mt_hash_nnz, sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_vr_nnz,      sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_vc_nnz,      sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_Mp_nnz,       &h_Mp_nnz,       sizeof(unsigned long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_Mt_hash_nnz,  &h_Mt_hash_nnz,  sizeof(unsigned long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_vr_nnz,       &h_vr_nnz,       sizeof(unsigned long long), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_vc_nnz,       &h_vc_nnz,       sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // classification kernel
    {util::Timer timer("PRRLDU (GPU) - classify kernel");
    eleClassify_kernel<<<grid_size, block_size>>>(    
        Ms_d_row_indices,   Ms_d_col_indices,   Ms_d_values,    Ms_nnz,   
        Mp_d_row_indices,   Mp_d_col_indices,   Mp_d_values,    d_Mp_nnz,                              
        Mt_hash_d_key,      Mt_hash_d_val,      d_Mt_hash_nnz,  Mt_col_num,                           
        vr_d_idx, vr_d_val, d_vr_nnz, vc_d_idx, vc_d_val, d_vc_nnz, s);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}

    // Return values
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_Mp_nnz,       d_Mp_nnz,       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_Mt_hash_nnz,  d_Mt_hash_nnz,  sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_vr_nnz,       d_vr_nnz,       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_vc_nnz,       d_vc_nnz,       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaFree(d_Mp_nnz));
    util::CHECK_CUDART_ERROR(cudaFree(d_Mt_hash_nnz));
    util::CHECK_CUDART_ERROR(cudaFree(d_vr_nnz));
    util::CHECK_CUDART_ERROR(cudaFree(d_vc_nnz));
    Mp_nnz = h_Mp_nnz;
    Mt_hash_nnz = h_Mt_hash_nnz;
    vr_nnz = h_vr_nnz;
    vc_nnz = h_vc_nnz;
    return;
}

// Note:  
// I found that the find and insert operations using the on-device reference of cuco::static_map  
// can be extremely slow for no apparent reason. Searching and inserting, for example,  
// 10,000 key-value pairs can take more than a minute.  
// What's worse, a fatal error is likely to occur when attempting to find, insert, and increment  
// simultaneously within a single kernel. The operations performed by cuco's on-device reference  
// do not appear to be ATOMIC!

// Outer-product update of Mt hash table on GPU (o1)
unsigned long long outerproduct_update_gpu(
    long long*& Mt_hash_d_key, double*& Mt_hash_d_val, long long& Mt_hash_nnz,               // In/Out: Hash table Mt
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,   // In: Sparse Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,   // In: Sparse Vc
    double const Mdenom, long long const M_col_num)                                          // In: Denominator and matrix column size
{
    util::Timer timer("PRRLDU (GPU) - Sparse phase A4");
    //*** Construct a Mt hash map ***//
    // Define key-value type
    using keyType = long long;
    using valType = double;

    // Empty slots are represented by reserved "sentinel" values. These values should be selected such that they never occur in your input data.
    keyType constexpr empty_key_sentinel   = -1;
    valType constexpr empty_value_sentinel = 0;
    
    // Compute capacity based on a 50% load factor
    auto constexpr load_factor = 0.5;
    std::size_t const hash_capacity = std::ceil(Mt_hash_nnz / load_factor);
    
    // Construct a hash map with "capacity" slots using -1 and 0 as the empty key/value sentinels.
    auto hash_map = cuco::static_map{hash_capacity,
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel},
        thrust::equal_to<keyType>{},
        cuco::linear_probing<1, cuco::default_hash_function<keyType>>{}};
    
    // CUCO references for insert/find
    //auto insert_find_ref = hash_map.ref(cuco::insert_and_find);
    //auto insert_ref = hash_map.ref(cuco::insert);
    auto find_ref = hash_map.ref(cuco::find);

    // Inserts all key-value pairs into the map. Using bulk insertion instead of the kernel implementation
    {util::Timer timer("PRRLDU (GPU) - hasht insert");
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(Mt_hash_d_key, Mt_hash_d_val));
    hash_map.insert(zipped, zipped + Mt_hash_nnz);}
    //insert_kernel<<<32,32>>>(insert_ref, Mt_hash_d_key, Mt_hash_d_val, Mt_hash_nnz, d_num_inserted);
    //util::CHECK_LAST_CUDART_ERROR();

    //*** Outer-product ***//   
    keyType* d_upd_hash_key; valType* d_upd_hash_val;
    {util::Timer timer("PRRLDU (GPU) - outer-prod");
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_upd_hash_key, sizeof(keyType) * (vc_nnz * vr_nnz))); 
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_upd_hash_val, sizeof(valType) * (vc_nnz * vr_nnz)));
    outer_product_kernel<<<grid_size, block_size>>>( d_upd_hash_key, d_upd_hash_val,                                              
        vr_d_idx, vr_d_val, vr_nnz, vc_d_idx, vc_d_val, vc_nnz, Mdenom, M_col_num);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}

    //*** Increment and Insertion ***//
    int64_t num_pairs = vr_nnz * vc_nnz;        // Number of key-value pairs produced by outer-product        
    keyType* d_insertKey; double* d_insertVal;  // New key-val pairs for insertion
    unsigned long long* d_num_inserted;         // Number of insertions
    unsigned long long h_num_inserted = 0;
    {util::Timer timer("PRRLDU (GPU) - A4verbose");
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_num_inserted,   sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMemset(d_num_inserted, 0, sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_insertKey, sizeof(keyType) * num_pairs));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_insertVal, sizeof(valType) * num_pairs));}

    // Increment and insertion kernel
    {util::Timer timer("PRRLDU (GPU) - hasht update");
    increment_values<<<grid_size, block_size>>>(find_ref, d_upd_hash_key, d_upd_hash_val, 
        num_pairs, d_insertKey, d_insertVal, d_num_inserted);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_num_inserted, d_num_inserted, sizeof(unsigned long long), cudaMemcpyDeviceToHost));}

    //*** Extract key-val pairs ***//
    keyType old_hash_size = hash_map.size();     // Old Ms/Mthash size
    Mt_hash_nnz = old_hash_size + h_num_inserted;  // New Ms/Mthash size
    
    {util::Timer timer("PRRLDU (GPU) - hashd retrieve");
    util::CHECK_CUDART_ERROR(cudaFree(Mt_hash_d_key));
    util::CHECK_CUDART_ERROR(cudaFree(Mt_hash_d_val));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Mt_hash_d_key, sizeof(keyType) * Mt_hash_nnz));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Mt_hash_d_val, sizeof(valType) * Mt_hash_nnz));
    hash_map.retrieve_all(Mt_hash_d_key, Mt_hash_d_val);
    util::CHECK_CUDART_ERROR(cudaMemcpy(Mt_hash_d_key + old_hash_size, d_insertKey, sizeof(keyType) * h_num_inserted, cudaMemcpyDeviceToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(Mt_hash_d_val + old_hash_size, d_insertVal, sizeof(valType) * h_num_inserted, cudaMemcpyDeviceToDevice));}

    // Memory release
    util::CHECK_CUDART_ERROR(cudaFree(d_upd_hash_key));
    util::CHECK_CUDART_ERROR(cudaFree(d_upd_hash_val));
    util::CHECK_CUDART_ERROR(cudaFree(d_num_inserted));
    util::CHECK_CUDART_ERROR(cudaFree(d_insertKey));
    util::CHECK_CUDART_ERROR(cudaFree(d_insertVal));
    return h_num_inserted;  // Return the number of insertion as a flag
}

// Outer-product update of Mt hash table on GPU (o2)
unsigned long long outerproduct_update_gpu_opt(
    long long*& Mt_hash_d_key, double*& Mt_hash_d_val, long long& Mt_hash_nnz,               // In/Out: Hash table Mt
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,   // In: Sparse Vr
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,   // In: Sparse Vc
    double const Mdenom, long long const M_col_num)                                          // In: Denominator and matrix column size
{
    util::Timer timer("PRRLDU (GPU) - Sparse phase A4");
    //--- Construct a Mt hash map ---//
    // Define key-value type
    using keyType = long long;
    using valType = double;

    // Empty slots are represented by reserved "sentinel" values. These values should be selected such that they never occur in your input data.
    keyType constexpr empty_key_sentinel   = -1;
    valType constexpr empty_value_sentinel = 0;
    
    // Compute capacity based on a 50% load factor
    auto constexpr load_factor = 0.5;
    std::size_t const hash_capacity = std::ceil((Mt_hash_nnz) / load_factor);
    
    // Construct a hash map with "capacity" slots using -1 and 0 as the empty key/value sentinels.
    auto hash_map = cuco::static_map{hash_capacity,
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel},
        thrust::equal_to<keyType>{},
        cuco::linear_probing<1, cuco::default_hash_function<keyType>>{}};
    
    // CUCO references for insert/find
    //auto insert_find_ref = hash_map.ref(cuco::insert_and_find);
    auto insert_ref = hash_map.ref(cuco::insert);
    auto find_ref = hash_map.ref(cuco::find);

    // Inserts all key-value pairs into the map. Using bulk insertion instead of the kernel implementation 
    {util::Timer timer("PRRLDU (GPU) - hasht insert");
    //auto zipped = thrust::make_zip_iterator(thrust::make_tuple(Mt_hash_d_key, Mt_hash_d_val));
    //hash_map.insert(zipped, zipped + Mt_hash_nnz);
    insert_kernel<<<grid_size, block_size>>>(insert_ref, Mt_hash_d_key, Mt_hash_d_val, Mt_hash_nnz);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}

    //--- Outer-product update ---//   
    int64_t num_pairs = vr_nnz * vc_nnz;        // Number of key-value pairs produced by outer-product        
    keyType* d_insertKey; double* d_insertVal;  // New key-val pairs for insertion
    unsigned long long* d_num_inserted;         // Number of insertions
    unsigned long long h_num_inserted = 0;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_num_inserted,   sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMemset(d_num_inserted, 0, sizeof(unsigned long long)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_insertKey, sizeof(keyType) * num_pairs));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_insertVal, sizeof(valType) * num_pairs));
    
    {util::Timer timer("PRRLDU (GPU) - outprod update");
    outerproduct_update_kernel<<<grid_size, block_size>>>(find_ref, d_insertKey, d_insertVal,   
        vr_d_idx, vr_d_val, vr_nnz, vc_d_idx, vc_d_val, vc_nnz, Mdenom, M_col_num, d_num_inserted); 
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_num_inserted, d_num_inserted, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    //--- Extract key-val pairs ---//
    keyType old_hash_size = hash_map.size();       // Old Ms/Mthash size
    Mt_hash_nnz = old_hash_size + h_num_inserted;  // New Ms/Mthash size
    
    {util::Timer timer("PRRLDU (GPU) - hashd retrieve");
    util::CHECK_CUDART_ERROR(cudaFree(Mt_hash_d_key));
    util::CHECK_CUDART_ERROR(cudaFree(Mt_hash_d_val));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Mt_hash_d_key, sizeof(keyType) * Mt_hash_nnz));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Mt_hash_d_val, sizeof(valType) * Mt_hash_nnz));
    hash_map.retrieve_all(Mt_hash_d_key, Mt_hash_d_val);
    util::CHECK_CUDART_ERROR(cudaMemcpy(Mt_hash_d_key + old_hash_size, d_insertKey, sizeof(keyType) * h_num_inserted, cudaMemcpyDeviceToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(Mt_hash_d_val + old_hash_size, d_insertVal, sizeof(valType) * h_num_inserted, cudaMemcpyDeviceToDevice));}

    // Memory release
    util::CHECK_CUDART_ERROR(cudaFree(d_num_inserted));
    util::CHECK_CUDART_ERROR(cudaFree(d_insertKey));
    util::CHECK_CUDART_ERROR(cudaFree(d_insertVal));
    return h_num_inserted;  // Return the number of insertion as a flag
}

// Convert Mt: hash key-val pairs to Ms: COO format 
void hash2COO_gpu(long long* Ms_d_row_indices, long long* Ms_d_col_indices, double* Ms_d_values, long long& Ms_nnz,                   // Out: To-be-updated Ms
    const long long* const Mt_hash_d_key, const double* const Mt_hash_d_val, long long const Mt_hash_nnz, long long const col_num)    // In: Mt hash key-val pairs
{
    util::Timer timer("PRRLDU (GPU) - Sparse phase A5");
    Ms_nnz = Mt_hash_nnz;
    hash2COO_kernel<<<grid_size, block_size>>>(
        Ms_d_row_indices, Ms_d_col_indices, Ms_d_values, col_num, 
        Mt_hash_d_key, Mt_hash_d_val, Mt_hash_nnz);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();
    return;
}

void perm_inv_gpu(long long* d_pivot_cols, long long* d_col_perm_inv, long long Nc)
{
    perm_inv_kernel<<<grid_size, block_size>>>(d_pivot_cols, d_col_perm_inv, Nc);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();
}

// Fusion kernel combining A2 (pivoting) and A3 (classification)
__global__ void fusion_kernel_A2A3(    
    long long* Ms_d_row_indices, long long* Ms_d_col_indices, double* Ms_d_values, long long const Ms_nnz,   
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long const Mp_nnz, unsigned long long* d_Mp_nnz,                               
    long long* Mt_hash_d_key, double* Mt_hash_d_val, unsigned long long* d_Mt_hash_nnz,                           
    long long* vr_d_idx, double* vr_d_val, unsigned long long* d_vr_nnz, 
    long long* vc_d_idx, double* vc_d_val, unsigned long long* d_vc_nnz,  
    long long const piv_r, long long const piv_c, long long const Mt_col_num, long long const s)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Pivot Mp
    while (tid < Mp_nnz) {
        if (Mp_d_row_indices[tid] == s)
            Mp_d_row_indices[tid] = piv_r;
        else if (Mp_d_row_indices[tid] == piv_r) 
            Mp_d_row_indices[tid] = s;
        if (Mp_d_col_indices[tid] == s)
            Mp_d_col_indices[tid] = piv_c;
        else if (Mp_d_col_indices[tid] == piv_c) 
            Mp_d_col_indices[tid] = s;            
        tid += blockDim.x * gridDim.x;
    }

    // TO BE MODIFIED HERE...
    tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Pivot Ms and classfication
    while (tid < Ms_nnz) {
        if (Ms_d_row_indices[tid] == s)
            Ms_d_row_indices[tid] = piv_r;
        else if (Ms_d_row_indices[tid] == piv_r) 
            Ms_d_row_indices[tid] = s;
        if (Ms_d_col_indices[tid] == s)
            Ms_d_col_indices[tid] = piv_c;
        else if (Ms_d_col_indices[tid] == piv_c) 
            Ms_d_col_indices[tid] = s;            
        
        long long r_i = Ms_d_row_indices[tid];
        long long c_i = Ms_d_col_indices[tid];
        double v_i = Ms_d_values[tid];
        if (fabs(v_i) > SPVAL_EPS) {
            if (r_i > s && c_i > s) {
                // Add the element to hash array
                unsigned long long old_Mt_hash_nnz = atomicAdd(d_Mt_hash_nnz, 1);           
                long long hashKey = cal_hashKey(r_i, c_i, Mt_col_num);
                Mt_hash_d_key[old_Mt_hash_nnz] = hashKey;
                Mt_hash_d_val[old_Mt_hash_nnz] = v_i;           
            } else {
                // Add the element to Mp
                unsigned long long old_Mp_nnz = atomicAdd(d_Mp_nnz, 1);
                Mp_d_row_indices[old_Mp_nnz] = r_i;
                Mp_d_col_indices[old_Mp_nnz] = c_i;
                Mp_d_values[old_Mp_nnz] = v_i;
                if (r_i != c_i) {
                    if (r_i == s) {
                        // Add the element to Vr
                        unsigned long long old_vr_nnz = atomicAdd(d_vr_nnz, 1);
                        vr_d_idx[old_vr_nnz] = c_i;
                        vr_d_val[old_vr_nnz] = v_i;
                    }
                    if (c_i == s) {
                        // Add the element to Vc
                        unsigned long long old_vc_nnz = atomicAdd(d_vc_nnz, 1);
                        vc_d_idx[old_vc_nnz] = r_i;
                        vc_d_val[old_vc_nnz] = v_i;
                    }
                }   
            }
        }
        
        tid += blockDim.x * gridDim.x;
    }
}

// Fusion kernel combining A2 (pivoting) and A3 (classification) using shared memory to optimize atomic writing
__global__ void fusion_kernel_A2A3_shm(    
    long long* Ms_d_row_indices, long long* Ms_d_col_indices, double* Ms_d_values, long long const Ms_nnz,   
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long const Mp_nnz, unsigned long long* d_Mp_nnz,                               
    long long* Mt_hash_d_key, double* Mt_hash_d_val, unsigned long long* d_Mt_hash_nnz,                           
    long long* vr_d_idx, double* vr_d_val, unsigned long long* d_vr_nnz, 
    long long* vc_d_idx, double* vc_d_val, unsigned long long* d_vc_nnz,  
    long long const piv_r, long long const piv_c, long long const Mt_col_num, long long const s)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Pivot Mp
    while (tid < Mp_nnz) {
        if (Mp_d_row_indices[tid] == s)
            Mp_d_row_indices[tid] = piv_r;
        else if (Mp_d_row_indices[tid] == piv_r) 
            Mp_d_row_indices[tid] = s;
        if (Mp_d_col_indices[tid] == s)
            Mp_d_col_indices[tid] = piv_c;
        else if (Mp_d_col_indices[tid] == piv_c) 
            Mp_d_col_indices[tid] = s;            
        tid += blockDim.x * gridDim.x;
    }

    // TO BE MODIFIED HERE...
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long stride = blockDim.x * gridDim.x;
    long long tx = threadIdx.x;
    unsigned long long old_idx;

    // Mt on shared memory
    __shared__ long long shm_Mt_hash_key[block_size];
    __shared__ double shm_Mt_hash_val[block_size];
    __shared__ unsigned int shm_Mt_nnz;
    __shared__ unsigned long long shm_old_Mt_hash_nnz;

    // Mp on shared memory
    __shared__ long long shm_Mp_row_idx[block_size];
    __shared__ long long shm_Mp_col_idx[block_size];
    __shared__ double shm_Mp_val[block_size];
    __shared__ unsigned int shm_Mp_nnz;
    __shared__ unsigned long long shm_old_Mp_nnz;

    // Pivot Ms and classfication
    while (tid < Ms_nnz) {
        if (Ms_d_row_indices[tid] == s)
            Ms_d_row_indices[tid] = piv_r;
        else if (Ms_d_row_indices[tid] == piv_r) 
            Ms_d_row_indices[tid] = s;
        if (Ms_d_col_indices[tid] == s)
            Ms_d_col_indices[tid] = piv_c;
        else if (Ms_d_col_indices[tid] == piv_c) 
            Ms_d_col_indices[tid] = s;            
        
        long long r_i = Ms_d_row_indices[tid];
        long long c_i = Ms_d_col_indices[tid];
        double v_i = Ms_d_values[tid];
        
        if (tx == 0) {
            shm_Mt_nnz = 0; // Initialize block-wise Mt nnz on shared memory
            shm_Mp_nnz = 0; // Initialize block-wise Mp nnz on shared memory
        }
        
        // Ensure all threads see the initialized value
        __syncthreads();
        
        if (fabs(v_i) > SPVAL_EPS) {
            if (r_i > s && c_i > s) {
                // Add the element to hash array
                long long hashKey = cal_hashKey(r_i, c_i, Mt_col_num);
                old_idx = atomicAdd(&shm_Mt_nnz, 1);
                shm_Mt_hash_key[old_idx] = hashKey;
                shm_Mt_hash_val[old_idx] = v_i;
            } else {
                // Add the element to Mp
                old_idx = atomicAdd(&shm_Mp_nnz, 1);
                shm_Mp_row_idx[old_idx] = r_i;
                shm_Mp_col_idx[old_idx] = c_i;
                shm_Mp_val[old_idx] = v_i;   
            }
        }

        __syncthreads();
        
        // Atomic write nnz on shared memory to global
        if (tx == 0) {
            shm_old_Mt_hash_nnz = atomicAdd(d_Mt_hash_nnz, shm_Mt_nnz);
            shm_old_Mp_nnz = atomicAdd(d_Mp_nnz, shm_Mp_nnz);
        }

        __syncthreads();

        // Write Mt on shared memory to global
        if (tx < shm_Mt_nnz) {
            Mt_hash_d_key[shm_old_Mt_hash_nnz + tx] = shm_Mt_hash_key[tx];
            Mt_hash_d_val[shm_old_Mt_hash_nnz + tx] = shm_Mt_hash_val[tx];
        }

        // Write Mp on shared memory to global / Also extract elements to Vr/Vc
        if (tx < shm_Mp_nnz) {
            r_i = shm_Mp_row_idx[tx];
            c_i = shm_Mp_col_idx[tx];
            v_i = shm_Mp_val[tx];
            Mp_d_row_indices[shm_old_Mp_nnz + tx] = r_i;
            Mp_d_col_indices[shm_old_Mp_nnz + tx] = c_i;
            Mp_d_values[shm_old_Mp_nnz + tx] = v_i;

            if (r_i != c_i && r_i == s) {
                // Add the element to Vr
                old_idx = atomicAdd(d_vr_nnz, 1);
                vr_d_idx[old_idx] = c_i;
                vr_d_val[old_idx] = v_i;
            } else if (r_i != c_i && c_i == s) {
                // Add the element to Vc
                old_idx = atomicAdd(d_vc_nnz, 1);
                vc_d_idx[old_idx] = r_i;
                vc_d_val[old_idx] = v_i;
            }
        }

        __syncthreads();

        tid += stride;
    }
}

// Fusion kernel combining A2 (pivoting) and A3 (classification) using shared memory to optimize atomic writing 
// (This version is different (in Mp handling) from version 1 and a little slower (to be investigated))
__global__ void fusion_kernel_A2A3_shm_2(    
    long long* Ms_d_row_indices, long long* Ms_d_col_indices, double* Ms_d_values, long long const Ms_nnz,   
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long const Mp_nnz, unsigned long long* d_Mp_nnz,                               
    long long* Mt_hash_d_key, double* Mt_hash_d_val, unsigned long long* d_Mt_hash_nnz,                           
    long long* vr_d_idx, double* vr_d_val, unsigned long long* d_vr_nnz, 
    long long* vc_d_idx, double* vc_d_val, unsigned long long* d_vc_nnz,  
    long long const piv_r, long long const piv_c, long long const Mt_col_num, long long const s)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Pivot Mp
    while (tid < Mp_nnz) {
        if (Mp_d_row_indices[tid] == s)
            Mp_d_row_indices[tid] = piv_r;
        else if (Mp_d_row_indices[tid] == piv_r) 
            Mp_d_row_indices[tid] = s;
        if (Mp_d_col_indices[tid] == s)
            Mp_d_col_indices[tid] = piv_c;
        else if (Mp_d_col_indices[tid] == piv_c) 
            Mp_d_col_indices[tid] = s;            
        tid += blockDim.x * gridDim.x;
    }

    // TO BE MODIFIED HERE...
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long stride = blockDim.x * gridDim.x;
    long long tx = threadIdx.x;
    unsigned long long old_idx;

    // Mt on shared memory
    __shared__ long long shm_Mt_hash_key[block_size];
    __shared__ double shm_Mt_hash_val[block_size];
    __shared__ unsigned int shm_Mt_nnz;
    __shared__ unsigned long long shm_old_Mt_hash_nnz;

    // Vr/Vc on shared memory
    __shared__ long long shm_vr_idx[block_size];
    __shared__ long long shm_vc_idx[block_size];
    __shared__ double shm_vr_val[block_size];
    __shared__ double shm_vc_val[block_size];
    __shared__ unsigned int shm_vr_nnz;
    __shared__ unsigned int shm_vc_nnz;
    __shared__ unsigned long long shm_old_vr_nnz;
    __shared__ unsigned long long shm_old_vc_nnz;
    __shared__ unsigned long long shm_old_Mp_nnz;

    // Pivot Ms and classfication
    while (tid < Ms_nnz) {
        if (Ms_d_row_indices[tid] == s)
            Ms_d_row_indices[tid] = piv_r;
        else if (Ms_d_row_indices[tid] == piv_r) 
            Ms_d_row_indices[tid] = s;
        if (Ms_d_col_indices[tid] == s)
            Ms_d_col_indices[tid] = piv_c;
        else if (Ms_d_col_indices[tid] == piv_c) 
            Ms_d_col_indices[tid] = s;            
        
        long long r_i = Ms_d_row_indices[tid];
        long long c_i = Ms_d_col_indices[tid];
        double v_i = Ms_d_values[tid];
        
        if (tx == 0) {
            shm_Mt_nnz = 0;  // Initialize block-wise Mt nnz on shared memory
            shm_vc_nnz = 0;  // Initialize block-wise vc nnz on shared memory
            shm_vr_nnz = 0;  // Initialize block-wise vr nnz on shared memory
        }
        
        // Ensure all threads see the initialized value
        __syncthreads();
        
        if (fabs(v_i) > SPVAL_EPS) {
            if (r_i > s && c_i > s) {
                // Add the element to hash array
                long long hashKey = cal_hashKey(r_i, c_i, Mt_col_num);
                old_idx = atomicAdd(&shm_Mt_nnz, 1);
                shm_Mt_hash_key[old_idx] = hashKey;
                shm_Mt_hash_val[old_idx] = v_i;
            } else {
                if (r_i == c_i) {
                    old_idx = atomicAdd(d_Mp_nnz, 1);
                    Mp_d_row_indices[old_idx] = r_i;
                    Mp_d_col_indices[old_idx] = c_i;
                    Mp_d_values[old_idx] = v_i;
                } else {
                    if (r_i == s) {
                        old_idx = atomicAdd(&shm_vr_nnz, 1);
                        shm_vr_idx[old_idx] = c_i;
                        shm_vr_val[old_idx] = v_i;
                    }
                    if (c_i == s) {
                        old_idx = atomicAdd(&shm_vc_nnz, 1);
                        shm_vc_idx[old_idx] = r_i;
                        shm_vc_val[old_idx] = v_i;
                    }
                }
            }
        }

        __syncthreads();
        
        // Atomic write nnz on shared memory to global
        if (tx == 0) {
            shm_old_Mt_hash_nnz = atomicAdd(d_Mt_hash_nnz, shm_Mt_nnz);
            shm_old_vr_nnz = atomicAdd(d_vr_nnz, shm_vr_nnz);
            shm_old_vc_nnz = atomicAdd(d_vc_nnz, shm_vc_nnz);
            shm_old_Mp_nnz = atomicAdd(d_Mp_nnz, shm_vr_nnz + shm_vc_nnz);
        }

        __syncthreads();

        // Write Mt on shared memory to global
        if (tx < shm_Mt_nnz) {
            unsigned long long offset_Mt = shm_old_Mt_hash_nnz + tx;
            Mt_hash_d_key[offset_Mt] = shm_Mt_hash_key[tx];
            Mt_hash_d_val[offset_Mt] = shm_Mt_hash_val[tx];
        }

        if (tx < shm_vr_nnz) {
            unsigned long long offset_vr = shm_old_vr_nnz + tx;
            unsigned long long offset_Mp = shm_old_Mp_nnz + tx;
            vr_d_idx[offset_vr] = shm_vr_idx[tx];
            vr_d_val[offset_vr] = shm_vr_val[tx];
            Mp_d_row_indices[offset_Mp] = s;
            Mp_d_col_indices[offset_Mp] = shm_vr_idx[tx];
            Mp_d_values[offset_Mp] = shm_vr_val[tx];
        }

        if (shm_vr_nnz <= tx && tx < (shm_vr_nnz + shm_vc_nnz)) {
            unsigned int offset = tx - shm_vr_nnz; 
            unsigned long long offset_vc = shm_old_vc_nnz + offset;
            unsigned long long offset_Mp = shm_old_Mp_nnz + tx;
            vc_d_idx[offset_vc] = shm_vc_idx[offset];
            vc_d_val[offset_vc] = shm_vc_val[offset];
            Mp_d_row_indices[offset_Mp] = shm_vc_idx[offset];
            Mp_d_col_indices[offset_Mp] = s;
            Mp_d_values[offset_Mp] = shm_vc_val[offset];
        }

        __syncthreads();

        tid += stride;
    }
}

// Fusion kernel combining A4 (outer-product update by hash table) and A5 (Ms update (only partial))
template <typename Map, typename KeyIter, typename ValIter>
__global__ void fusion_kernel_A4A5(Map map_ref, KeyIter* Ms_d_row_indices, KeyIter* Ms_d_col_indices, ValIter* Ms_d_values,       
    const long long* const vr_d_idx, const double* const vr_d_val, long long const vr_nnz,    
    const long long* const vc_d_idx, const double* const vc_d_val, long long const vc_nnz,      
    double const Mdenom, long long const col_num, unsigned long long* num_inserted)             
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < (vr_nnz * vc_nnz)) {
        // Indices of vc/vr
        long long vc_idx = tid / vr_nnz;
        long long vr_idx = tid % vr_nnz;
        
        // Get the hash key-value pair used to update hash table
        KeyIter upd_hash_key = cal_hashKey(vc_d_idx[vc_idx], vr_d_idx[vr_idx], col_num); 
        ValIter upd_hash_val = -1.0 * vc_d_val[vc_idx] * vr_d_val[vr_idx] / Mdenom;

        // Search the hash map
        auto found = map_ref.find(upd_hash_key);
        if (found != map_ref.end()) {
            // If the key exists, atomically increment the associated value
            auto ref = cuda::atomic_ref<typename Map::mapped_type, cuda::thread_scope_device>{found->second};
            ref.fetch_add(upd_hash_val, cuda::memory_order_relaxed);
        } else {
            // If the key does not exist, append/insert the new key-val pair directly to the device array for key-val pairs
            unsigned long long old_idx = atomicAdd(num_inserted, 1);
            Ms_d_row_indices[old_idx] = upd_hash_key / col_num;
            Ms_d_col_indices[old_idx] = upd_hash_key % col_num;
            Ms_d_values[old_idx] = upd_hash_val;
        }

        // Next grid
        tid += blockDim.x * gridDim.x;
    }
}

// Fusion of kernel A2 A3 A4 and A5
unsigned long long A2345_fusion( 
    long long*& Ms_d_row_indices, long long*& Ms_d_col_indices, double*& Ms_d_values, long long& Ms_nnz, long long& Ms_capacity,   
    long long* Mp_d_row_indices, long long* Mp_d_col_indices, double* Mp_d_values, long long& Mp_nnz,                            
    long long* Mt_hash_d_key, double* Mt_hash_d_val, long long& Mt_hash_nnz,                           
    long long* vr_d_idx, double* vr_d_val, long long& vr_nnz, 
    long long* vc_d_idx, double* vc_d_val, long long& vc_nnz,  
    long long const piv_r, long long const piv_c, long long const M_col_num, double const Mdenom, long long const s)
{
    util::Timer timer("PRRLDU (GPU) - Fusion A2+3+4+5");
    using keyType = long long;
    using nnzType = unsigned long long;
    using valType = double;

    // nnz object memory management
    nnzType h_Mp_nnz = Mp_nnz, h_Mt_hash_nnz = 0, h_vr_nnz = 0, h_vc_nnz = 0;
    nnzType* d_Mp_nnz, *d_Mt_hash_nnz, *d_vr_nnz, *d_vc_nnz;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_Mp_nnz,      sizeof(nnzType))); 
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_Mt_hash_nnz, sizeof(nnzType)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_vr_nnz,      sizeof(nnzType)));
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_vc_nnz,      sizeof(nnzType)));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_Mp_nnz,      &h_Mp_nnz,      sizeof(nnzType), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_Mt_hash_nnz, &h_Mt_hash_nnz, sizeof(nnzType), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_vr_nnz,      &h_vr_nnz,      sizeof(nnzType), cudaMemcpyHostToDevice));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_vc_nnz,      &h_vc_nnz,      sizeof(nnzType), cudaMemcpyHostToDevice));
 
    // A2 & A3 fusion kernel
    #ifdef SHMEM_OPT
    {util::Timer timer("PRRLDU (GPU) - Fusion ker1 shm");
    fusion_kernel_A2A3_shm<<<grid_size, block_size>>>(    
        Ms_d_row_indices, Ms_d_col_indices, Ms_d_values, Ms_nnz,   
        Mp_d_row_indices, Mp_d_col_indices, Mp_d_values, Mp_nnz, d_Mp_nnz,                               
        Mt_hash_d_key, Mt_hash_d_val, d_Mt_hash_nnz,                           
        vr_d_idx, vr_d_val, d_vr_nnz, 
        vc_d_idx, vc_d_val, d_vc_nnz,  
        piv_r, piv_c, M_col_num, s);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}
    #else
    {util::Timer timer("PRRLDU (GPU) - Fusion ker1");
    fusion_kernel_A2A3<<<grid_size, block_size>>>(    
        Ms_d_row_indices, Ms_d_col_indices, Ms_d_values, Ms_nnz,   
        Mp_d_row_indices, Mp_d_col_indices, Mp_d_values, Mp_nnz, d_Mp_nnz,                               
        Mt_hash_d_key, Mt_hash_d_val, d_Mt_hash_nnz,                           
        vr_d_idx, vr_d_val, d_vr_nnz, 
        vc_d_idx, vc_d_val, d_vc_nnz,  
        piv_r, piv_c, M_col_num, s);
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}
    #endif
 
    // Post A3 & A3: Update nnz objects
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_Mp_nnz,      d_Mp_nnz,      sizeof(nnzType), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_Mt_hash_nnz, d_Mt_hash_nnz, sizeof(nnzType), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_vr_nnz,      d_vr_nnz,      sizeof(nnzType), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_vc_nnz,      d_vc_nnz,      sizeof(nnzType), cudaMemcpyDeviceToHost));
    util::CHECK_CUDART_ERROR(cudaFree(d_Mp_nnz));
    util::CHECK_CUDART_ERROR(cudaFree(d_Mt_hash_nnz));
    util::CHECK_CUDART_ERROR(cudaFree(d_vr_nnz));
    util::CHECK_CUDART_ERROR(cudaFree(d_vc_nnz));
    Mp_nnz = h_Mp_nnz;
    Mt_hash_nnz = h_Mt_hash_nnz;
    vr_nnz = h_vr_nnz;
    vc_nnz = h_vc_nnz;

    // Post A3 & A3: Reconstruct Ms space
    nnzType num_pairs = vr_nnz * vc_nnz;  // Number of key-value pairs produced by outer-product        
    nnzType* d_num_inserted;              // Number of insertions
    nnzType h_num_inserted = 0;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_num_inserted,   sizeof(nnzType)));
    util::CHECK_CUDART_ERROR(cudaMemset(d_num_inserted, 0, sizeof(nnzType)));

    // Reallocation of Ms is very slow - to be investigated
    Ms_capacity = Mt_hash_nnz + num_pairs;
    util::CHECK_CUDART_ERROR(cudaFree(Ms_d_row_indices));
    util::CHECK_CUDART_ERROR(cudaFree(Ms_d_col_indices));
    util::CHECK_CUDART_ERROR(cudaFree(Ms_d_values));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Ms_d_row_indices, sizeof(long long) * Ms_capacity));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Ms_d_col_indices, sizeof(long long) * Ms_capacity));
    util::CHECK_CUDART_ERROR(cudaMalloc(&Ms_d_values, sizeof(double) * Ms_capacity));
    
    // CUCO references for insert/find
    // Hash table initialization
    keyType constexpr empty_key_sentinel   = -1;
    valType constexpr empty_value_sentinel = 0;

    auto constexpr load_factor = 0.5;     // Compute capacity based on a 50% load factor
    std::size_t const hash_capacity = std::ceil(Mt_hash_nnz / load_factor);

    auto hash_map = cuco::static_map{hash_capacity,
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel},
        thrust::equal_to<keyType>{},
        cuco::linear_probing<1, cuco::default_hash_function<keyType>>{}};
    
    auto insert_ref = hash_map.ref(cuco::insert);  // insert reference 
    auto find_ref = hash_map.ref(cuco::find);      // search reference

    {util::Timer timer("PRRLDU (GPU) - hasht insert");
    // Inserts all key-value pairs into the map. Using bulk insertion instead of the kernel implementation
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(Mt_hash_d_key, Mt_hash_d_val));
    hash_map.insert(zipped, zipped + Mt_hash_nnz);
    util::CHECK_LAST_CUDART_ERROR();}
        
    {util::Timer timer("PRRLDU (GPU) - Fusion ker2");
    fusion_kernel_A4A5<<<grid_size, block_size>>>(find_ref, Ms_d_row_indices, Ms_d_col_indices, Ms_d_values,   
        vr_d_idx, vr_d_val, vr_nnz, vc_d_idx, vc_d_val, vc_nnz, Mdenom, M_col_num, d_num_inserted); 
    cudaDeviceSynchronize();
    util::CHECK_LAST_CUDART_ERROR();}
    util::CHECK_CUDART_ERROR(cudaMemcpy(&h_num_inserted, d_num_inserted, sizeof(nnzType), cudaMemcpyDeviceToHost));
    
    //--- Extract key-val pairs ---//
    keyType old_hash_size = hash_map.size();       // Old Ms/Mthash size
    Mt_hash_nnz = old_hash_size + h_num_inserted;  // New Ms/Mthash size
    Ms_nnz = Mt_hash_nnz;
        
    {util::Timer timer("PRRLDU (GPU) - hashd retrieve");
    hash_map.retrieve_all(Mt_hash_d_key, Mt_hash_d_val);}

    hash2COO_kernel<<<grid_size, block_size>>>(
    Ms_d_row_indices + h_num_inserted, Ms_d_col_indices + h_num_inserted, Ms_d_values + h_num_inserted, M_col_num, 
    Mt_hash_d_key, Mt_hash_d_val, old_hash_size);

    util::CHECK_CUDART_ERROR(cudaFree(d_num_inserted));
    return h_num_inserted;  // Return the number of insertion as a flag
}