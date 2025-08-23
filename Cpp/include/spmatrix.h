// spmatrix.h - Sparse matrix toolkit
#ifndef SPMATRIX_H
#define SPMATRIX_H

#include "core.h"
#include <mkl/mkl_spblas.h>

// A simple sparse vector implementation
template <typename T>
struct SparseVector {
    long long size;  // dimension
    long long nnz;   // number of non-zeros
    long long capacity;
    long long* idx;  // indices array
    T* val;        // values array
    
    SparseVector(long long n, long long capacity) : size(n), nnz(0), capacity(capacity) {
        idx = new long long[capacity];
        val = new T[capacity];
    }
    
    ~SparseVector() {
        delete[] idx;
        delete[] val;
    }
    
    void resize(long long newCap) {
        long long* newIdx = new long long[newCap];
        T* newVal = new T[newCap]; 
        std::copy(idx, idx + nnz, newIdx);
        std::copy(val, val + nnz, newVal);
        delete[] idx;
        delete[] val;
        idx = newIdx;
        val = newVal;
        capacity = newCap;
    }
    
    void set(long long i, T v) {
        if(std::abs(v) > 1e-14) {
            if(nnz >= capacity) 
                resize(capacity * 2);
            idx[nnz] = i;
            val[nnz] = v;
            nnz++;
        }
    }

    void print() {
        for(long long i = 0; i < nnz; i++) {
            std::cout << "idx[" << i << "]=" << idx[i] 
                    << ", val[" << i << "]=" << val[i] << std::endl;
        }
    }
};

// COO-format sparse matrix using hash map on CPU
template<typename T>
struct COOMatrix_l1 {
private:
    long long rows, cols;
    std::unordered_map<long long, T> data;  // hash table

    // 2D coordinates -> 1D key for hash table
    long long getKey(long long row, long long col) const {
        long long key = row * cols + col;
        if (key < 0)
            std::cout << "NOTE! KEY BUG!" << std::endl;
        return key;
    }

public:
    // Default constructor
    COOMatrix_l1(long long r, long long c) : rows(r), cols(c) {}

    // Set key-value pair
    void set(long long row, long long col, T val) {
       if (row >= rows || col >= cols) throw std::out_of_range("Index out of bounds!");
       long long key = getKey(row, col);
       if (std::abs(val) > 1e-14) {
           data[key] = val;
       } else {
           data.erase(key);
       }
    }

    T get(long long row, long long col) const {
       long long key = getKey(row, col);
       auto it = data.find(key);
       return it != data.end() ? it->second : 0;
    }

    void addUpdate(long long row, long long col, T val) {
        if (row >= rows || col >= cols) throw std::out_of_range("Index out of bounds!");
        if (std::abs(val) > 1e-14) {
            long long key = getKey(row, col);
            auto it = data.find(key);
            if (it != data.end())
                data[key] = data[key] + val;
            else
                data[key] = val;
        } 
    }

    // Iterator-style access to non-zero elements
    const std::unordered_map<long long, T>& getRawData() const {  
        return data;
    }

    // Helper method to convert key back to coordinates
    std::pair<long long, long long> getCoordinates(long long key) const {
        return {key / cols, key % cols};
    }

    long long nnz() const {
        return data.size();
    }

    void print() const {
        for(long long i = 0; i < rows; i++) {
            for(long long j = 0; j < cols; j++) {
                std::cout << get(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

// COO-format matrix (l2), a compact sparse format for algorithms on CPU 
template<typename T>
struct COOMatrix_l2 {
    long long rows;  // Number of rows in the full matrix
    long long cols;  // Number of columns in the full matrix
    long long capacity;   // Maximum capacity for non-zero elements  
    long long nnz_count;  // Current number of non-zero elements 
    
    // Arrays to store the non-zero elements and their positions
    long long* row_indices;   // Row indices of non-zero elements
    long long* col_indices;   // Column indices of non-zero elements
    T* values;              // Values of non-zero elements

    // Constructor
    COOMatrix_l2()
        : rows(0), cols(0), capacity(0), nnz_count(0) {
        row_indices = nullptr;
        col_indices = nullptr;
        values = nullptr;
    }
    
    // Constructor
    COOMatrix_l2(long long num_rows, long long num_cols, long long initial_capacity = 100) 
        : rows(num_rows), cols(num_cols), capacity(initial_capacity), nnz_count(0) {
        row_indices = new long long[capacity];
        col_indices = new long long[capacity];
        values = new T[capacity];
    }

    // Destructor
    ~COOMatrix_l2() {
        if (row_indices != nullptr)
            delete[] row_indices;
        if (col_indices != nullptr)
            delete[] col_indices;
        if (values != nullptr)
            delete[] values;
    }

    // Copy constructor
    COOMatrix_l2(const COOMatrix_l2& other) 
        : rows(other.rows), cols(other.cols), 
          capacity(other.capacity), nnz_count(other.nnz_count) {
        row_indices = new long long[capacity];
        col_indices = new long long[capacity];
        values = new T[capacity];
        
        std::memcpy(row_indices, other.row_indices, nnz_count * sizeof(long long));
        std::memcpy(col_indices, other.col_indices, nnz_count * sizeof(long long));
        std::memcpy(values, other.values, nnz_count * sizeof(T));
    }

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

    // Re-constructor
    void reconst(long long num_rows, long long num_cols, long long initial_capacity = 100) {
        rows = num_rows;
        cols = num_cols;
        capacity = initial_capacity;
        nnz_count = 0;

        if (row_indices != nullptr) delete[] row_indices;
        if (col_indices != nullptr) delete[] col_indices;
        if (values != nullptr) delete[] values;
    
        row_indices = new long long[capacity];
        col_indices = new long long[capacity];
        values = new T[capacity];
        return;
    }

    // Add a non-zero element to the matrix
    void add_element(long long row, long long col, T value) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        if (std::abs(value) > 1e-14) {  // Only store non-zero values
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            row_indices[nnz_count] = row;
            col_indices[nnz_count] = col;
            values[nnz_count] = value;
            nnz_count++;
        }
    }

    // Get the value at a specific position
    T get(long long row, long long col) const {
        for (long long i = 0; i < nnz_count; ++i) {
            if (row_indices[i] == row && col_indices[i] == col) {
                return values[i];
            }
        }
        return T(0);  // Return zero if element not found
    }

    // Update the value at a specific position
    void update(long long row, long long col, T val) {
        if (std::abs(val) > 1e-14) {
            for (long long i = 0; i < nnz_count; ++i) {
                if (row_indices[i] == row && col_indices[i] == col) {
                    values[i] = val; 
                    return;                
                }
            }
            add_element(row, col, val);
        }
    }

    // Update the value by +=
    void addUpdate(long long row, long long col, T val) {
        if (std::abs(val) > 1e-14) {
            for (long long i = 0; i < nnz_count; ++i) {
                if (row_indices[i] == row && col_indices[i] == col) {
                    values[i] += val; 
                    return;                
                }
            }
            add_element(row, col, val);
        }
    }

    // Full format
    T* todense() const {
        T* full = new T[rows * cols]{0};
        for (long long i = 0; i < nnz_count; ++i) {
            full[row_indices[i] * cols + col_indices[i]] = values[i];
        }
        return full;
    }

    // In-place reshape
    void reshape(long long new_row, long long new_col) {
        if (new_row * new_col != rows * cols) {
            throw std::runtime_error("New shape does not match with the original dimension!");
        }
        // Reshape
        long long idx;
        for (long long i = 0; i < nnz_count; ++i) {
            idx = row_indices[i] * cols + col_indices[i];
            row_indices[i] = idx / new_col;
            col_indices[i] = idx % new_col;
        }
        rows = new_row;
        cols = new_col;
        return;    
    }

    // Get number of non-zero elements
    long long nnz() const {
        return nnz_count;
    }

    // Sort elements by row and column indices
    void sort() {
        long long* indices = new long long[nnz_count];
        for (long long i = 0; i < nnz_count; ++i) {
            indices[i] = i;
        }

        std::sort(indices, indices + nnz_count,
            [this](long long i1, long long i2) {
                if (row_indices[i1] != row_indices[i2])
                    return row_indices[i1] < row_indices[i2];
                return col_indices[i1] < col_indices[i2];
            });

        // Create temporary arrays for sorting
        long long* new_row_indices = new long long[capacity];
        long long* new_col_indices = new long long[capacity];
        T* new_values = new T[capacity];

        for (long long i = 0; i < nnz_count; ++i) {
            new_row_indices[i] = row_indices[indices[i]];
            new_col_indices[i] = col_indices[indices[i]];
            new_values[i] = values[indices[i]];
        }

        // Swap pointers
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
        delete[] indices;

        row_indices = new_row_indices;
        col_indices = new_col_indices;
        values = new_values;
    }

    // Print the matrix in COO format
    void print() const {
        std::cout << "COO Matrix (" << rows << " x " << cols << "), "
                  << nnz_count << " non-zero elements:\n";
        for (long long i = 0; i < nnz_count; ++i) {
            std::cout << "(" << row_indices[i] << ", " << col_indices[i] 
                      << ") = " << values[i] << "\n";
        }
    }

    // Release the memory explicitly
    void explicit_destroy() {
        if (row_indices != nullptr) {
            delete[] row_indices;
            row_indices = nullptr;
        }
        if (col_indices != nullptr) {
            delete[] col_indices;
            col_indices = nullptr;
        }
        if (values != nullptr) {
            delete[] values;
            values = nullptr;
        }
        nnz_count = 0;
    }

    // Solve the upper triangular system (N*N)
    // Assume the matrix is an upper triangular matrix) 
    // Assume the diagonal entries are all 1
    void utrsv(long long N, T* b) {
        if (N > rows || N > cols) 
            throw std::runtime_error("N could not be larger than row/column!");        
        
        // A very naive implementation of sparse trsv
        for (long long i = 0; i < N; ++i) {
            T temp = 0.0;
            for (long long j = 0; j < i; ++j) {
                temp += get(N-1-i, N-1-j) * b[N-1-j];
            }
            //b[N-1-i] = (b[N-1-i] - temp) / get(N-1-i, N-1-i);
            b[N-1-i] = b[N-1-i] - temp;
        }

        return;
    }

    // Select sub-rows of the sparse COO matrix
    COOMatrix_l2<T> subrow(const long long* pivot_rows, long long rank) {
        // Initialize the resulting sparse matrix
        COOMatrix_l2<T> result(rank, cols);

        // Select rows from the row-pivot array
        for (long long i = 0; i < nnz_count; ++i) {
            long long row_idx = row_indices[i];
            for (long long j = 0; j < rank; ++j) {
                long long prow_idx = pivot_rows[j];
                if (prow_idx == row_idx) {
                    result.add_element(j, col_indices[i], values[i]);
                }
            }
        }

        return result;
    }

    // Select sub-columns of the sparse COO matrix
    COOMatrix_l2<T> subcol(const long long* pivot_cols, long long rank) {
        // Initialize the resulting sparse matrix
        COOMatrix_l2<T> result(rows, rank);
        
        // Select columns from the column-pivot array
        for (long long i = 0; i < nnz_count; ++i) {
            long long col_idx = col_indices[i];
            for (long long j = 0; j < rank; ++j) {
                long long pcol_idx = pivot_cols[j];
                if (pcol_idx == col_idx) {
                    result.add_element(row_indices[i], j, values[i]);
                }
            }    
        }
        
        return result;
    }

    COOMatrix_l2<T> multiply(const COOMatrix_l2<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }

        // Initialize result matrix
        COOMatrix_l2<T> result(rows, other.cols);
        
        // Create a map to accumulate results for each position
        std::map<std::pair<long long, long long>, T> temp_results;
        
        // For each non-zero element in the first matrix
        for (long long i = 0; i < nnz_count; ++i) {
            long long row_a = row_indices[i];
            T val_a = values[i];
            
            // For each non-zero element in the second matrix
            for (long long j = 0; j < other.nnz_count; ++j) {
                // Only multiply if the column of first matches row of second
                if (col_indices[i] == other.row_indices[j]) {
                    long long col_b = other.col_indices[j];
                    T val_b = other.values[j];
                    
                    // Accumulate the product
                    temp_results[{row_a, col_b}] += val_a * val_b;
                }
            }
        }
        
        // Convert accumulated results to COO format
        for (const auto& entry : temp_results) {
            if (entry.second != T(0)) {  // Only store non-zero results
                result.add_element(entry.first.first, entry.first.second, entry.second);
            }
        }
        
        // Sort the result for better access patterns
        result.sort();
        return result;
    }

    // Generate random non-zero entries
    void generate_random(double density, unsigned int seed, T min_val = T(1), T max_val = T(100)) {
        if (density < 0.0 || density > 1.0) {
            throw std::invalid_argument("Density must be between 0 and 1");
        }

        // Calculate number of non-zero elements based on density
        long long total_elements = rows * cols;
        long long target_nnz = static_cast<long long>(density * total_elements);
        
        // Clear existing data
        explicit_destroy();
        
        // Initialize with new capacity
        capacity = target_nnz;
        row_indices = new long long[capacity];
        col_indices = new long long[capacity];
        values = new T[capacity];
        
        // Set up random number generators
        std::mt19937 gen(seed);
        std::uniform_int_distribution<long long> row_dist(0, rows - 1);
        std::uniform_int_distribution<long long> col_dist(0, cols - 1);
        
        // For floating point values
        std::uniform_real_distribution<double> val_dist(
            static_cast<double>(min_val), 
            static_cast<double>(max_val)
        );
        
        // Use set to ensure unique positions
        std::set<std::pair<long long, long long>> positions;
        
        // Generate unique random positions
        while (positions.size() < target_nnz) {
            long long row = row_dist(gen);
            long long col = col_dist(gen);
            positions.insert({row, col});
        }
        
        // Fill the matrix with random values at these positions
        nnz_count = 0;
        for (const auto& pos : positions) {
            T value;
            if constexpr (std::is_integral<T>::value) {
                // For integer types
                value = static_cast<T>(std::round(val_dist(gen)));
            } else {
                // For floating point types
                value = static_cast<T>(val_dist(gen));
            }
            
            row_indices[nnz_count] = pos.first;
            col_indices[nnz_count] = pos.second;
            values[nnz_count] = value;
            nnz_count++;
        }
        
        // Sort the entries
        sort();
    }
};

#endif