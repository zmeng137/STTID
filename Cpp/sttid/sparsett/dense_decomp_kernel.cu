#include "structures.h"
#include "cutil.h"
#include "gpu_kernel.h"

// CUDA kernel to find max absolute value in a submatrix
__global__ void maxAbsSubmatrix(const float* M, float* result, int matrix_width, int submatrix_start) {
    // Shared memory for block reduction
    __shared__ float sdata[256];  // Assuming block size <= 256
    
    // Global thread index
    unsigned int tid = threadIdx.x;
    
    // Initialize shared memory to 0
    sdata[tid] = 0.0f;
    
    // Calculate indices for the submatrix
    int row = blockIdx.y * blockDim.y + threadIdx.y + submatrix_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x + submatrix_start;
    
    float max_val = 0.0f;
    
    // Check if thread is within submatrix bounds
    if (row < matrix_width && col < matrix_width) {
        // Calculate 1D index in the flattened matrix
        int idx = row * matrix_width + col;
        
        // Get absolute value
        max_val = fabsf(M[idx]);
    }
    
    // Store in shared memory
    sdata[tid] = max_val;
    
    __syncthreads();
    
    // Perform parallel reduction to find maximum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block's result
    if (tid == 0) {
        // Use atomicMax with float values via float-to-int bit conversion
        // This is needed because CUDA doesn't have a native atomicMax for floats
        float old = *result;
        float val = sdata[0];
        while (val > old) {
            float assumed = old;
            old = atomicCAS((int*)result, 
                          __float_as_int(assumed),
                          __float_as_int(val));
            old = __int_as_float(old);
            if (old == assumed)
                break;
        }
    }
}

__global__ void dense_outprod_kernel(double* d_M_full, long long Nc, long long s, long long upd_size)
{
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < upd_size) {
        long long i = tid / (Nc - (s+1)) + s + 1;
        long long j = tid % (Nc - (s+1)) + s + 1;
        double outprod = d_M_full[i * Nc + s] * d_M_full[s * Nc + j] / d_M_full[s * Nc + s];
        d_M_full[i * Nc + j] = d_M_full[i * Nc + j] - outprod;
        tid += gridDim.x * blockDim.x;
    }

}

void denseLU_gpukernel(double* M_full, const long long Nr, const long long Nc, 
    long long& s, const long long k, const double cutoff, double& inf_error, 
    long long* rps, long long* cps, decompRes::SparsePrrlduRes<double> resultSet)
{
    double* d_M_full;
    util::CHECK_CUDART_ERROR(cudaMalloc(&d_M_full, sizeof(double) * Nr * Nc));
    util::CHECK_CUDART_ERROR(cudaMemcpy(d_M_full, M_full, sizeof(double) * Nr * Nc, cudaMemcpyHostToDevice));

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
        std::cout << "s=" << s << ", Mabs_max=" << Mabs_max << ", pivrc=" << piv_c << "," << piv_c;
        if (Mabs_max < cutoff) {
            std::cout << "Terminate!" << std::endl;
            inf_error = Mabs_max;
            break;
        }

        // Update diagonal entries
        resultSet.d[s] = M_full[piv_r * Nc + piv_c]; 

        // Row/Column swap
        //util::CHECK_CUDART_ERROR(cudaStreamSynchronize(0));     
        cublasHandle_t handle;
        util::CHECK_CUBLAS_ERROR(cublasCreate(&handle));
        cublasDswap_64(handle, Nc, d_M_full + piv_r * Nc, 1, d_M_full + s * Nc, 1);
        cublasDswap_64(handle, Nr, d_M_full + piv_c, Nc, d_M_full + s, Nc);
        util::CHECK_CUBLAS_ERROR(cublasDestroy(handle));
               
        // Outer-product update 
        if (s < k - 1) {
            long long upd_size = (Nr - (s+1)) * (Nc - (s+1));
            cudaDeviceSynchronize();
            dense_outprod_kernel<<<128, 128>>>(d_M_full, Nc, s, upd_size);
            cudaDeviceSynchronize();
            util::CHECK_LAST_CUDART_ERROR();
        }
        util::CHECK_CUDART_ERROR(cudaMemcpy(M_full, d_M_full, sizeof(double) * Nr * Nc, cudaMemcpyDeviceToHost));

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

    util::CHECK_CUDART_ERROR(cudaFree(d_M_full));
}