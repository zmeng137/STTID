cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR) # Minimum cmake version 
project(STTID CUDA CXX)  # Project name

# Function to copy executable to a second location after build
function(add_executable_dual_output target_name source_files)
    add_executable(${target_name} ${source_files})
    
    # Primary output directory: STTID/src/build/exe/
    set_target_properties(${target_name} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/exe)
    
    # Copy to secondary location after build: STTID/exe/
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy 
            $<TARGET_FILE:${target_name}> 
            ${CMAKE_SOURCE_DIR}/../exe/$<TARGET_FILE_NAME:${target_name}>)
endfunction()

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "G++ version: ${GXX_VERSION}")

# Set default build type
set(CMAKE_BUILD_TYPE Release)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Include directory for headers
include_directories(include)
include(cmake/CPM.cmake)

# Find libraries
find_package(CUDAToolkit REQUIRED)          # CUDA Toolkit
message(STATUS "CUDA version: ${CUDA_VERSION}")  # Output CUDA version and installation path
message(STATUS "CUDA installation path: ${CUDA_TOOLKIT_ROOT_DIR}")

# Add CCCL first (dependency CCCL -> CUCO) using CPM
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY NVIDIA/cccl
  GIT_TAG main
)

# Add CUCO using CPM 
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

# Shared library for MKL compute subroutines
set(SHARED_SOURCES_MKLKERNEL 
    sttid/mkl_kernel/dense_kernel.cpp
    sttid/mkl_kernel/trsv_kernel.cpp
    sttid/util/timer.cpp)

set(SHARED_SOURCES_MKLKERNEL_LS 
    sttid/mkl_kernel/dense_kernel.cpp
    sttid/largesize/trsv_kernel_largesize.cpp
    sttid/util/timer.cpp)
  
set(SHARED_SOURCES_CUSPARSE
    sttid/cusp_kernel/cusparse_kernel.cu
    sttid/util/timer.cpp)

# Shared library for sparse tensor-train network
set (SHARED_SOURCES_SPTT_LS_CPU
    sttid/largesize/spinterpolative_l1_largesize.cpp
    sttid/largesize/spinterpolative_l2_largesize.cpp
    sttid/largesize/spinterpolative_l3_largesize.cpp
    sttid/largesize/sptensortrain_largesize.cpp
    sttid/util/timer.cpp
)

set (SHARED_SOURCES_SPTT_LS_GPU
    sttid/sparsett/decomp_kernel.cu
    sttid/sparsett/dense_decomp_kernel.cu
    sttid/largesize/spinterpolative_l1_largesize.cpp
    sttid/largesize/spinterpolative_l2_largesize.cpp
    sttid/largesize/spinterpolative_l3_largesize.cpp
    sttid/largesize/spinterpolative_l3g_largesize.cu
    sttid/largesize/sptensortrain_largesize.cpp
    sttid/util/timer.cpp
)

# MKL compute kernel
add_library(shared_mkl_kernel STATIC ${SHARED_SOURCES_MKLKERNEL})
target_compile_definitions(shared_mkl_kernel PRIVATE MKL_ILP64 _GLIBCXX_USE_CXX17_ABI=1)  # Define required compiler definitions
target_link_libraries(shared_mkl_kernel PRIVATE -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

add_library(shared_mkl_kernel_ls STATIC ${SHARED_SOURCES_MKLKERNEL_LS})
target_compile_definitions(shared_mkl_kernel_ls PRIVATE MKL_ILP64 _GLIBCXX_USE_CXX17_ABI=1)  # Define required compiler definitions
target_link_libraries(shared_mkl_kernel_ls PRIVATE -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

# CUSPARSE compute kernel
add_library(shared_cusparse_kernel STATIC ${SHARED_SOURCES_CUSPARSE})
target_link_libraries(shared_cusparse_kernel PRIVATE CUDA::cudart CUDA::cusparse)
target_compile_definitions(shared_cusparse_kernel PRIVATE MKL_ILP64 _GLIBCXX_USE_CXX17_ABI=1)  # Define required compiler definitions
target_link_libraries(shared_cusparse_kernel PRIVATE -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

# STTID lib
add_library(shared_lib_sparseTT_ls_cpu STATIC ${SHARED_SOURCES_SPTT_LS_CPU})
add_library(shared_lib_sparseTT_ls_gpu STATIC ${SHARED_SOURCES_SPTT_LS_GPU})

target_link_libraries(shared_lib_sparseTT_ls_cpu PRIVATE shared_mkl_kernel_ls)

target_compile_definitions(shared_lib_sparseTT_ls_gpu PRIVATE ENABLE_GPU)
target_link_libraries(shared_lib_sparseTT_ls_gpu PRIVATE CUDA::cudart CUDA::cublas cuco shared_cusparse_kernel)
target_link_libraries(shared_lib_sparseTT_ls_gpu PRIVATE shared_mkl_kernel_ls)

# Define sources
set(SOURCES_SYNS4 sttid/synthetic_test/sparse_test_dim4.cpp)
set(SOURCES_SYNS5 sttid/synthetic_test/sparse_test_dim5.cpp)

# Executables for executable tests
add_executable_dual_output(sttid_dim4_cpu ${SOURCES_SYNS4})
add_executable_dual_output(sttid_dim4_gpu ${SOURCES_SYNS4})
add_executable_dual_output(sttid_dim5_cpu ${SOURCES_SYNS5})
add_executable_dual_output(sttid_dim5_gpu ${SOURCES_SYNS5})

# Link libraries for test exe
target_link_libraries(sttid_dim4_cpu PRIVATE shared_lib_sparseTT_ls_cpu CUDA::cudart)
target_link_libraries(sttid_dim4_gpu PRIVATE shared_lib_sparseTT_ls_gpu CUDA::cudart)
target_link_libraries(sttid_dim5_cpu PRIVATE shared_lib_sparseTT_ls_cpu CUDA::cudart)
target_link_libraries(sttid_dim5_gpu PRIVATE shared_lib_sparseTT_ls_gpu CUDA::cudart)