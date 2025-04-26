cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR) # Minimum cmake version 
project(SparseTensorTrain CUDA CXX)  # Project name

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "G++ version: ${GXX_VERSION}")

# Set default build type
set(CMAKE_BUILD_TYPE Debug)

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
    src/mkl_kernel/dense_kernel.cpp
    src/mkl_kernel/trsv_kernel.cpp
    src/util/timer.cpp)

set(SHARED_SOURCES_MKLKERNEL_LS 
    src/mkl_kernel/dense_kernel.cpp
    src/largesize/trsv_kernel_largesize.cpp
    src/util/timer.cpp)
  
set(SHARED_SOURCES_CUSPARSE
    src/cusp_kernel/cusparse_kernel.cu
    src/util/timer.cpp)

# Shared library for sparse tensor-train network
set (SHARED_SOURCES_SPTT
    src/sparsett/spinterpolative_l1.cpp
    src/sparsett/spinterpolative_l2.cpp
    src/sparsett/spinterpolative_l3.cpp
    src/sparsett/decomp_kernel.cu
    src/sparsett/dense_decomp_kernel.cu
    src/sparsett/spinterpolative_l3g.cu
    src/sparsett/sptensortrain.cpp
    src/util/timer.cpp
)

set (SHARED_SOURCES_SPTT_LS
    src/sparsett/decomp_kernel.cu
    src/sparsett/dense_decomp_kernel.cu
    src/largesize/spinterpolative_l3_largesize.cpp
    src/largesize/spinterpolative_l3g_largesize.cu
    src/largesize/sptensortrain_largesize.cpp
    src/util/timer.cpp
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

add_library(shared_lib_sparseTT STATIC ${SHARED_SOURCES_SPTT})
add_library(shared_lib_sparseTT_ls STATIC ${SHARED_SOURCES_SPTT_LS})
#target_compile_options(shared_lib_spcuTT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

target_link_libraries(shared_lib_sparseTT PRIVATE CUDA::cudart CUDA::cublas cuco shared_cusparse_kernel)
target_link_libraries(shared_lib_sparseTT PRIVATE shared_mkl_kernel)
target_link_libraries(shared_lib_sparseTT_ls PRIVATE CUDA::cudart CUDA::cublas cuco shared_cusparse_kernel)
target_link_libraries(shared_lib_sparseTT_ls PRIVATE shared_mkl_kernel_ls)

# Define sources
set(SOURCES_SYNS4 src/synthetic_test/sparse_test_dim4.cpp)
set(SOURCES_SYNS5 src/synthetic_test/sparse_test_dim5.cpp)

# Executables for synthetic tests
add_executable(synthetic_sparse_test_dim4 ${SOURCES_SYNS4})
add_executable(synthetic_sparse_test_dim5 ${SOURCES_SYNS5})
add_executable(synthetic_sparse_test_dim4_ls ${SOURCES_SYNS4})
add_executable(synthetic_sparse_test_dim5_ls ${SOURCES_SYNS5})

# Link libraries for synthetic test exe
target_link_libraries(synthetic_sparse_test_dim4 PRIVATE shared_lib_sparseTT CUDA::cudart)
target_link_libraries(synthetic_sparse_test_dim5 PRIVATE shared_lib_sparseTT CUDA::cudart)
target_link_libraries(synthetic_sparse_test_dim4_ls PRIVATE shared_lib_sparseTT_ls CUDA::cudart)
target_link_libraries(synthetic_sparse_test_dim5_ls PRIVATE shared_lib_sparseTT_ls CUDA::cudart)
set_target_properties(synthetic_sparse_test_dim4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim5 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim4_ls PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim5_ls PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)