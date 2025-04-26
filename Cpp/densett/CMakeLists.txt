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

# Shared library for dense tensor-train network
set(SHARED_SOURCES_DENSETT
  decomposition.cpp
  interpolative.cpp
  tensortrain.cpp
  subroutines.cpp
  timer.cpp
)

# Dense TT (using tblis, blas, lapack)
add_library(shared_lib_denseTT STATIC ${SHARED_SOURCES_DENSETT})
target_link_libraries(shared_lib_denseTT PUBLIC ${TBLIS_LIB} ${BLAS_LIBRARIES} ${LAPACKE_LIB})

# Define sources
set(SOURCES_SYND dense_test.cpp)

# Executables for synthetic tests
add_executable(synthetic_dense_test ${SOURCES_SYND})

# Link libraries for synthetic test exe
target_link_libraries(synthetic_dense_test PRIVATE shared_lib_denseTT)

# Output executables to different directories
set_target_properties(synthetic_dense_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
