#ifndef SPMAT_DEVICE_H
#define SPMAT_DEVICE_H

#include "cutil.h"

// A simple sparse vector implementation
template <typename T>
struct SparseVector_device {
    long long size;     // dimension
    long long nnz;      // number of non-zeros
    long long capacity; // Capacity
    long long* d_idx;   // indices array
    T* d_val;   // values array
    
    SparseVector_device(long long n, long long capacity) : size(n), nnz(0), capacity(capacity) {
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_idx, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_val, sizeof(T) * capacity)); 
    }
    
    ~SparseVector_device() {
        util::CHECK_CUDART_ERROR(cudaFree(d_idx));
        util::CHECK_CUDART_ERROR(cudaFree(d_val));
    }
    
    void resize(long long newCap) {
        long long* d_newIdx;
        T* d_newVal; 
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_newIdx, sizeof(long long) * newCap));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_newVal, sizeof(T) * newCap));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_newIdx, d_idx, nnz * sizeof(long long), cudaMemcpyDeviceToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_newVal, d_val, nnz * sizeof(T), cudaMemcpyDeviceToDevice));
        util::CHECK_CUDART_ERROR(cudaFree(d_idx));
        util::CHECK_CUDART_ERROR(cudaFree(d_val));
        d_idx = d_newIdx;
        d_val = d_newVal;
        capacity = newCap;
    }
    
    /*
    void set(long long i, T v) {
        if(std::abs(v) > SPVALEPS) {
            if(nnz >= capacity) 
                resize(capacity * 2);
            idx[nnz] = i;
            val[nnz] = v;
            nnz++;
        }
    }

    void print() {
        //...
        for(long long i = 0; i < nnz; i++) {
            std::cout << "idx[" << i << "]=" << idx[i] 
                    << ", val[" << i << "]=" << val[i] << std::endl;
        }
    }*/
};

// COO-format matrix (l2) on device, a compact sparse format for algorithms on CPU 
template<typename T>
struct COOMatrix_l2_device {
    long long rows;       // Number of rows in the full matrix
    long long cols;       // Number of columns in the full matrix
    long long capacity;   // Maximum capacity for non-zero elements  
    long long nnz_count;  // Current number of non-zero elements 
    
    // Arrays to store the non-zero elements and their positions
    long long* d_row_indices;   // Row indices of non-zero elements
    long long* d_col_indices;   // Column indices of non-zero elements
    T* d_values;        // Values of non-zero elements

    // Null constructor
    COOMatrix_l2_device()
        : rows(0), cols(0), capacity(0), nnz_count(0) {
        d_row_indices = nullptr;
        d_col_indices = nullptr;
        d_values = nullptr;
    }
    
    // Allocation constructor
    COOMatrix_l2_device(long long num_rows, long long num_cols, long long initial_capacity = 100) 
        : rows(num_rows), cols(num_cols), capacity(initial_capacity), nnz_count(0) {
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_row_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_col_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_values, sizeof(T) * capacity));
    }

    // Re-constructor
    void reconst(long long num_rows, long long num_cols, long long initial_capacity = 100) {
        rows = num_rows;
        cols = num_cols;
        capacity = initial_capacity;
        nnz_count = 0;

        if (d_row_indices != nullptr) util::CHECK_CUDART_ERROR(cudaFree(d_row_indices));
        if (d_col_indices != nullptr) util::CHECK_CUDART_ERROR(cudaFree(d_col_indices));
        if (d_values != nullptr) util::CHECK_CUDART_ERROR(cudaFree(d_values));
    
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_row_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_col_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_values, sizeof(T) * capacity));
        return;
    }

    // Implicit destructor
    ~COOMatrix_l2_device() {
        if (d_row_indices != nullptr)
            util::CHECK_CUDART_ERROR(cudaFree(d_row_indices));
        if (d_col_indices != nullptr)
            util::CHECK_CUDART_ERROR(cudaFree(d_col_indices));
        if (d_values != nullptr)
            util::CHECK_CUDART_ERROR(cudaFree(d_values));
    }

    // Explicit memory release
    void explicit_destroy() {
        if (d_row_indices != nullptr) {
            util::CHECK_CUDART_ERROR(cudaFree(d_row_indices));
            d_row_indices = nullptr;
        }
        if (d_col_indices != nullptr) {
            util::CHECK_CUDART_ERROR(cudaFree(d_col_indices));
            d_col_indices = nullptr;
        }
        if (d_values != nullptr) {
            util::CHECK_CUDART_ERROR(cudaFree(d_values));
            d_values = nullptr;
        }
        nnz_count = 0;
        capacity = 0;
    }

    // Get number of non-zero elements
    long long nnz() const {
        return nnz_count;
    }

    // Copy constructor host -> device
    COOMatrix_l2_device(const COOMatrix_l2<T>& other) 
        : rows(other.rows), cols(other.cols), capacity(other.capacity), nnz_count(other.nnz_count) {
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_row_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_col_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_values, sizeof(T) * capacity));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_row_indices, other.row_indices, nnz_count * sizeof(long long), cudaMemcpyHostToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_col_indices, other.col_indices, nnz_count * sizeof(long long), cudaMemcpyHostToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_values, other.values, nnz_count * sizeof(long long), cudaMemcpyHostToDevice));
    }

    // Copy constructor device -> device
    COOMatrix_l2_device(const COOMatrix_l2_device& other) 
        : rows(other.rows), cols(other.cols), capacity(other.capacity), nnz_count(other.nnz_count) {
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_row_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_col_indices, sizeof(long long) * capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&d_values, sizeof(T) * capacity));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_row_indices, other.d_row_indices, nnz_count * sizeof(long long), cudaMemcpyDeviceToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_col_indices, other.d_col_indices, nnz_count * sizeof(long long), cudaMemcpyDeviceToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(d_values, other.d_values, nnz_count * sizeof(long long), cudaMemcpyDeviceToDevice));
    }

    // Resize arrays when capacity is reached
    void resize(long long new_capacity) {
        if (new_capacity <= capacity) {
            return;
        } 
    
        long long* new_d_row_indices, *new_d_col_indices;
        T* new_d_values;
        util::CHECK_CUDART_ERROR(cudaMalloc(&new_d_row_indices, sizeof(long long) * new_capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&new_d_col_indices, sizeof(long long) * new_capacity));
        util::CHECK_CUDART_ERROR(cudaMalloc(&new_d_values, sizeof(T) * new_capacity));

        util::CHECK_CUDART_ERROR(cudaMemcpy(new_d_row_indices, d_row_indices, nnz_count * sizeof(long long), cudaMemcpyDeviceToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(new_d_col_indices, d_col_indices, nnz_count * sizeof(long long), cudaMemcpyDeviceToDevice));
        util::CHECK_CUDART_ERROR(cudaMemcpy(new_d_values, d_values, nnz_count * sizeof(T), cudaMemcpyDeviceToDevice));

        util::CHECK_CUDART_ERROR(cudaFree(d_row_indices));
        util::CHECK_CUDART_ERROR(cudaFree(d_col_indices));
        util::CHECK_CUDART_ERROR(cudaFree(d_values));
        
        d_row_indices = new_d_row_indices;
        d_col_indices = new_d_col_indices;
        d_values = new_d_values;
        capacity = new_capacity;
    }

/*
    // Assignment operator
    COOMatrix_l2& operator=(const COOMatrix_l2& other) {
        if (this != &other) {
            // Free existing resources
            delete[] row_indices;
            delete[] col_indices;
            delete[] values;
            
            // Copy new data
            rows = other.rows;
            cols = other.cols;
            capacity = other.capacity;
            nnz_count = other.nnz_count;
            
            row_indices = new long long[capacity];
            col_indices = new long long[capacity];
            values = new T[capacity];
            
            std::memcpy(row_indices, other.row_indices, nnz_count * sizeof(long long));
            std::memcpy(col_indices, other.col_indices, nnz_count * sizeof(long long));
            std::memcpy(values, other.values, nnz_count * sizeof(T));
        }
        return *this;
    }

    // Resize arrays when capacity is reached
    void resize(long long new_capacity) {
        long long* new_row_indices = new long long[new_capacity];
        long long* new_col_indices = new long long[new_capacity];
        T* new_values = new T[new_capacity];
        
        std::memcpy(new_row_indices, row_indices, nnz_count * sizeof(long long));
        std::memcpy(new_col_indices, col_indices, nnz_count * sizeof(long long));
        std::memcpy(new_values, values, nnz_count * sizeof(T));
        
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
        
        row_indices = new_row_indices;
        col_indices = new_col_indices;
        values = new_values;
        capacity = new_capacity;
    }

    // Add a non-zero element to the matrix
    void add_element(long long row, long long col, T value) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        if (std::abs(value) > SPVALEPS) {  // Only store non-zero values
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            row_indices[nnz_count] = row;
            col_indices[nnz_count] = col;
            values[nnz_count] = value;
            nnz_count++;
        }
    }
*/
};

#endif