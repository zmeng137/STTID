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
include_directories(../include)
include_directories(include)

# Find libraries
find_package(BLAS REQUIRED)                 # BLAS (Note the collision between OpenBLAS and MKL BLAS)
find_package(LAPACK REQUIRED)               # LAPACK
find_library(LAPACKE_LIB lapacke REQUIRED)  # LAPACKE
find_library(TBLIS_LIB tblis REQUIRED)      # TBLIS (for fast dense tensor computation)
find_package(CUDAToolkit REQUIRED)          # CUDA Toolkit
message(STATUS "CUDA version: ${CUDA_VERSION}")  # Output CUDA version and installation path
message(STATUS "CUDA installation path: ${CUDA_TOOLKIT_ROOT_DIR}")

# Shared library for dense tensor-train network
set(SHARED_SOURCES_DENSETT
  ../densett/decomposition.cpp
  ../densett/interpolative.cpp
  ../densett/tensortrain.cpp
  ../densett/subroutines.cpp
  ../densett/timer.cpp
)

set(SHARED_SOURCES_CUSPARSE
    ../src/cusp_kernel/cusparse_kernel.cu
    ../src/util/timer.cpp)

# Dense TT (using tblis, blas, lapack)
add_library(shared_lib_denseTT STATIC ${SHARED_SOURCES_DENSETT})
target_link_libraries(shared_lib_denseTT PUBLIC ${TBLIS_LIB} ${BLAS_LIBRARIES} ${LAPACKE_LIB})

add_library(shared_cusparse_kernel STATIC ${SHARED_SOURCES_CUSPARSE})
target_link_libraries(shared_cusparse_kernel PRIVATE CUDA::cudart CUDA::cusparse)
target_compile_definitions(shared_cusparse_kernel PRIVATE MKL_ILP64 _GLIBCXX_USE_CXX17_ABI=1)  # Define required compiler definitions
target_link_libraries(shared_cusparse_kernel PRIVATE -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

# Define sources
set(SOURCES_UT1 unit_test_1.cpp)
set(SOURCES_UT2 unit_test_2.cpp)
set(SOURCES_UT3 unit_test_3.cpp)
set(SOURCES_UT4 unit_test_4.cpp)
set(SOURCES_UT5 unit_test_5.cpp)
set(SOURCES_UT6 unit_test_6.cpp)

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
#add_executable(unit_test_5 ${SOURCES_UT5})
add_executable(unit_test_6 ${SOURCES_UT6})

# Link libraries for unit test exe
target_link_libraries(unit_test_1 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_2 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_3 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_4 PRIVATE shared_lib_denseTT gtest_main)
#target_link_libraries(unit_test_5 PRIVATE shared_lib_sparseTT gtest_main)
target_link_libraries(unit_test_6 PRIVATE CUDA::cudart CUDA::cusparse shared_cusparse_kernel gtest_main)

# Output executables to different directories
set_target_properties(unit_test_1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
#set_target_properties(unit_test_5 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_6 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
