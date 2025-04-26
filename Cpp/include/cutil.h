#ifndef CUTIL_CUH
#define CUTIL_CUH

#include <stdio.h>
#include <chrono>
#include <thread>
#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <cuda_runtime.h> // CUDA Runtime
#include <cusparse.h>     // CUSPARSE 
#include <cublas_v2.h>    // CUBLAS (v2!)

#define CSRT 1
#define CSCT 0

namespace util{

inline void getCuversion()
{
    int cublasVersion;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasGetVersion(cublasHandle, &cublasVersion);
    printf("cuBLAS version: %d\n", cublasVersion);
    cublasDestroy(cublasHandle);

    // Check cuSPARSE version
    int cusparseVersion;
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseGetVersion(cusparseHandle, &cusparseVersion);
    printf("cuSPARSE version: %d\n", cusparseVersion);
    cusparseDestroy(cusparseHandle);
}

// CUDA runtime-error detection function
#define CHECK_CUDART_ERROR(val) checkCudaRt((val), #val, __FILE__, __LINE__)
inline void checkCudaRt(cudaError_t err, const char* const func, const char* const file, int const line)
{
	if (err != cudaSuccess) 
		fprintf(stderr, "CUDA Runtime Error at: %s: %d\n %s %s\n", file, line, cudaGetErrorString(err), func);
}

// CUDA last runtime-error detection function
#define CHECK_LAST_CUDART_ERROR() checkLastCudaRt(__FILE__, __LINE__)
inline void checkLastCudaRt(const char* const file, int const line)
{
	cudaError_t const err{ cudaGetLastError() };
	if (err != cudaSuccess)
		fprintf(stderr, "CUDA Runtime Error at: %s: %d\n %s \n", file, line, cudaGetErrorString(err));
}

// CUSPARSE error detection function
#define CHECK_CUSPARSE_ERROR(val) checkCusparse((val), #val, __FILE__, __LINE__)
inline void checkCusparse(cusparseStatus_t err, const char* const func, const char* const file, int const line) 
{
    if (err != CUSPARSE_STATUS_SUCCESS) 
		fprintf(stderr, "CUSPARSE Error at: %s: %d\n %s %s\n", file, line, cusparseGetErrorName(err), func);
}

// CUBLAS error detection function
#define CHECK_CUBLAS_ERROR(val) checkCublas((val), #val, __FILE__, __LINE__)
inline void checkCublas(cublasStatus_t err, const char* const func, const char* const file, int const line) 
{
    if (err != CUBLAS_STATUS_SUCCESS) 
		fprintf(stderr, "CUBLAS Error at: %s: %d\n %s %s\n", file, line, cublasGetStatusName(err), func);
}

class DeviceMemoryLogger
{
private:
    std::ofstream log_file;
    std::thread logger_thread;
    bool running = true;
    int sample_interval_ms;
    int device_id = 0; // Default GPU device

    // Get current timestamp as string
    std::string get_timestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_r(&now_time, &tm_buf);

        char buffer[25];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
        return std::string(buffer);
    }

    void logging_loop()
    {
        // Write CSV header
        log_file << "Timestamp,Used (MB),Free (MB),Total (MB)" << std::endl;

        while (running)
        {
            size_t free_byte;
            size_t total_byte;

            cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

            if (cuda_status != cudaSuccess)
            {
                log_file << get_timestamp() << ",ERROR: " << cudaGetErrorString(cuda_status) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
                continue;
            }

            // Calculate used memory
            size_t used_byte = total_byte - free_byte;

            // Convert bytes to MB for readability
            double used_mb = used_byte / 1048576.0; // 1024*1024
            double free_mb = free_byte / 1048576.0;
            double total_mb = total_byte / 1048576.0;

            // Write to CSV file
            log_file << get_timestamp() << ","
                     << std::fixed << std::setprecision(2) << used_mb << ","
                     << free_mb << ","
                     << total_mb << std::endl;

            // Ensure data is written to disk
            log_file.flush();

            std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
        }
    }

public:
    DeviceMemoryLogger(const std::string &filename, int interval_ms = 1000, int gpu_id = 0)
        : sample_interval_ms(interval_ms), device_id(gpu_id)
    {

        // Set the GPU device
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess)
        {
            std::cerr << "Error setting CUDA device: " << cudaGetErrorString(error) << std::endl;
            return;
        }

        // Check if CUDA is available
        int device_count;
        error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0)
        {
            std::cerr << "No CUDA devices available" << std::endl;
            return;
        }

        log_file.open(filename);
        if (!log_file.is_open())
        {
            std::cerr << "Failed to open log file: " << filename << std::endl;
            return;
        }

        // Get device properties for log header
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        log_file << "# GPU Memory Logger" << std::endl;
        log_file << "# Device: " << prop.name << std::endl;
        log_file << "# Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        log_file << "# Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        log_file << "# Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;

        logger_thread = std::thread(&DeviceMemoryLogger::logging_loop, this);
        std::cout << "GPU Memory Logger started for device " << device_id << ": " << prop.name << std::endl;
    }

    ~DeviceMemoryLogger()
    {
        running = false;
        if (logger_thread.joinable())
        {
            logger_thread.join();
        }
        if (log_file.is_open())
        {
            log_file.close();
        }
    }

    // Get current GPU memory info
    static void printCurrentMemoryUsage(int dev_id = 0)
    {
        cudaSetDevice(dev_id);

        size_t free_byte;
        size_t total_byte;

        cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
        if (cuda_status != cudaSuccess)
        {
            std::cerr << "Error getting memory info: " << cudaGetErrorString(cuda_status) << std::endl;
            return;
        }

        size_t used_byte = total_byte - free_byte;

        std::cout << "GPU Memory Usage:" << std::endl;
        std::cout << "  Used:  " << used_byte / 1048576.0 << " MB" << std::endl;
        std::cout << "  Free:  " << free_byte / 1048576.0 << " MB" << std::endl;
        std::cout << "  Total: " << total_byte / 1048576.0 << " MB" << std::endl;
    }
};

}

#endif