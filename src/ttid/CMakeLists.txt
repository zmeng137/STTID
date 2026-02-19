cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR) # Minimum cmake version 
project(DTTID CXX)  # Project name

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
            ${CMAKE_SOURCE_DIR}/../../exe/$<TARGET_FILE_NAME:${target_name}>)
endfunction()

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "G++ version: ${GXX_VERSION}")
set(CMAKE_BUILD_TYPE Release)

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
set(SOURCES_SYND_4D dense_test_4d.cpp)
set(SOURCES_SYND_5D dense_test_5d.cpp)
set(SOURCES_RECD_4D dense_recon_4d.cpp)

# Executables for synthetic tests
add_executable_dual_output(ttid_dim4_cpu ${SOURCES_SYND_4D})
add_executable_dual_output(ttid_dim5_cpu ${SOURCES_SYND_5D})
add_executable_dual_output(recon_dense_test_4d ${SOURCES_RECD_4D})

# Link libraries for synthetic test exe
target_link_libraries(ttid_dim4_cpu PRIVATE shared_lib_denseTT)
target_link_libraries(ttid_dim5_cpu PRIVATE shared_lib_denseTT)
target_link_libraries(recon_dense_test_4d PRIVATE shared_lib_denseTT)