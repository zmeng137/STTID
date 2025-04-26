cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR) # Minimum cmake version 

project(SparseTensorTrain CUDA CXX)  # Project name

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "G++ version: ${GXX_VERSION}")

# Set default build type
#set(default_build_type "Debug")
set(CMAKE_BUILD_TYPE Debug)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Include directory for headers
include_directories(include)
include(cmake/CPM.cmake)

# Find libraries
find_package(BLAS REQUIRED)                 # BLAS (Note the collision between OpenBLAS and MKL BLAS)
find_package(LAPACK REQUIRED)               # LAPACK
find_library(LAPACKE_LIB lapacke REQUIRED)  # LAPACKE
find_library(TBLIS_LIB tblis REQUIRED)      # TBLIS (for fast dense tensor computation)
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

# Shared library for dense tensor-train network
set(SHARED_SOURCES_DENSETT
  src/densett/decomposition.cpp
  src/densett/interpolative.cpp
  src/densett/tensortrain.cpp
  src/densett/subroutines.cpp
  src/util/timer.cpp
)

# Shared library for MKL compute subroutines
set(SHARED_SOURCES_MKLKERNEL 
    src/mkl_kernel/dense_kernel.cpp
    src/mkl_kernel/trsv_kernel.cpp
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
    src/sparsett/spinterpolative_l1.cpp
    src/sparsett/spinterpolative_l2.cpp
    src/sparsett/spinterpolative_l3_largesize.cpp
    src/sparsett/decomp_kernel.cu
    src/sparsett/dense_decomp_kernel.cu
    src/sparsett/spinterpolative_l3g_largesize.cu
    src/sparsett/sptensortrain_largesize.cpp
    src/util/timer.cpp
)

# Dense TT (using tblis, blas, lapack)
add_library(shared_lib_denseTT STATIC ${SHARED_SOURCES_DENSETT})
target_link_libraries(shared_lib_denseTT PUBLIC ${TBLIS_LIB} ${BLAS_LIBRARIES} ${LAPACKE_LIB})

# MKL compute kernel
add_library(shared_mkl_kernel STATIC ${SHARED_SOURCES_MKLKERNEL})
target_compile_definitions(shared_mkl_kernel PRIVATE MKL_ILP64 _GLIBCXX_USE_CXX17_ABI=1)  # Define required compiler definitions
target_link_libraries(shared_mkl_kernel PRIVATE -Wl,--no-as-needed -lmkl_intel_ilp64  # Link MKL
                      -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

# CUSPARSE compute kernel
add_library(shared_cusparse_kernel STATIC ${SHARED_SOURCES_CUSPARSE})
target_link_libraries(shared_cusparse_kernel PRIVATE CUDA::cudart CUDA::cusparse)
target_compile_definitions(shared_cusparse_kernel PRIVATE MKL_ILP64 _GLIBCXX_USE_CXX17_ABI=1)  # Define required compiler definitions
target_link_libraries(shared_cusparse_kernel PRIVATE -Wl,--no-as-needed -lmkl_intel_ilp64  # Link MKL
                      -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

add_library(shared_lib_sparseTT STATIC ${SHARED_SOURCES_SPTT})
add_library(shared_lib_sparseTT_ls STATIC ${SHARED_SOURCES_SPTT_LS})
#target_compile_options(shared_lib_spcuTT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

target_link_libraries(shared_lib_sparseTT PRIVATE CUDA::cudart CUDA::cublas cuco shared_cusparse_kernel)
target_link_libraries(shared_lib_sparseTT PRIVATE shared_mkl_kernel)
target_link_libraries(shared_lib_sparseTT_ls PRIVATE CUDA::cudart CUDA::cublas cuco shared_cusparse_kernel)
target_link_libraries(shared_lib_sparseTT_ls PRIVATE shared_mkl_kernel)

# Define sources
set(SOURCES_SYND src/synthetic_test/dense_test.cpp)
set(SOURCES_SYNS4 src/synthetic_test/sparse_test_dim4.cpp)
set(SOURCES_SYNS5 src/synthetic_test/sparse_test_dim5.cpp)
set(SOURCES_UT1 src/unit_test/unit_test_1.cpp)
set(SOURCES_UT2 src/unit_test/unit_test_2.cpp)
set(SOURCES_UT3 src/unit_test/unit_test_3.cpp)
set(SOURCES_UT4 src/unit_test/unit_test_4.cpp)
set(SOURCES_UT5 src/unit_test/unit_test_5.cpp)
set(SOURCES_UT6 src/unit_test/unit_test_6.cpp)

# Executables for synthetic tests
add_executable(synthetic_dense_test ${SOURCES_SYND})
add_executable(synthetic_sparse_test_dim4 ${SOURCES_SYNS4})
add_executable(synthetic_sparse_test_dim5 ${SOURCES_SYNS5})
add_executable(synthetic_sparse_test_dim4_ls ${SOURCES_SYNS4})
add_executable(synthetic_sparse_test_dim5_ls ${SOURCES_SYNS5})

# Link libraries for synthetic test exe
target_link_libraries(synthetic_dense_test PRIVATE shared_lib_denseTT)
target_link_libraries(synthetic_sparse_test_dim4 PRIVATE shared_lib_sparseTT CUDA::cudart)
target_link_libraries(synthetic_sparse_test_dim5 PRIVATE shared_lib_sparseTT CUDA::cudart)
target_link_libraries(synthetic_sparse_test_dim4_ls PRIVATE shared_lib_sparseTT_ls CUDA::cudart)
target_link_libraries(synthetic_sparse_test_dim5_ls PRIVATE shared_lib_sparseTT_ls CUDA::cudart)

# Google Test setup
# If Google Test is not installed, download and build it
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# Prevent GoogleTest from building its own tests
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()  # Enable testing in CMake

# Executables for unit tests
add_executable(unit_test_1 ${SOURCES_UT1})
add_executable(unit_test_2 ${SOURCES_UT2})
add_executable(unit_test_3 ${SOURCES_UT3})
add_executable(unit_test_4 ${SOURCES_UT4})
add_executable(unit_test_5 ${SOURCES_UT5})
add_executable(unit_test_6 ${SOURCES_UT6})

# Link libraries for unit test exe
target_link_libraries(unit_test_1 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_2 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_3 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_4 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_5 PRIVATE shared_lib_sparseTT gtest_main)
target_link_libraries(unit_test_6 PRIVATE CUDA::cudart CUDA::cusparse shared_cusparse_kernel gtest_main)

# Output executables to different directories
set_target_properties(unit_test_1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_5 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_6 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(synthetic_dense_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim5 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim4_ls PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test_dim5_ls PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)