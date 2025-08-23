// sptensor.h - Sparse tensor toolkit
#ifndef SPTENSOR_H
#define SPTENSOR_H

#include <tblis/tblis.h>
#include "structures.h"
#include "util.h"

// Helper for index sequence
template<size_t... Is>
struct IndexSequence {};

template<size_t N, size_t... Is>
struct MakeIndexSequence : MakeIndexSequence<N-1, N-1, Is...> {};

template<size_t... Is>
struct MakeIndexSequence<0, Is...> {
    using type = IndexSequence<Is...>;
};

template<typename T, size_t Order>
class COOTensor {
public:
    // Array to store dimensions
    std::array<size_t, Order> dimensions;

    // Maximum capacity for non-zero elements
    size_t capacity;
    
    // Current number of non-zero elements
    size_t nnz_count;
    
    // Arrays to store indices and values
    std::array<size_t*, Order> indices;  // Array of pointers to index arrays
    T* values;                           // Values of non-zero elements

    // Helper function to check if indices are within bounds
    template<size_t... Is>
    bool check_bounds(const std::array<size_t, Order>& idx, IndexSequence<Is...>) const {
        return ((idx[Is] < dimensions[Is]) && ...);
    }

    // Helper function for lexicographic comparison
    bool compare_indices(size_t i1, size_t i2) const {
        for (size_t dim = 0; dim < Order; ++dim) {
            if (indices[dim][i1] != indices[dim][i2]) {
                return indices[dim][i1] < indices[dim][i2];
            }
        }
        return false;
    }
    
    // Helper struct for contraction specification
    struct ContractionDims {
        size_t this_dim;   // Dimension from this tensor
        size_t other_dim;  // Dimension from other tensor
        ContractionDims(size_t t, size_t o) : this_dim(t), other_dim(o) {}
    };

    // Constructor with empty init
    COOTensor()
        : capacity(0), nnz_count(0) {
            
    }

    // Constructor with dimensions
    template<typename... Dims>
    COOTensor(size_t initial_capacity, Dims... dims) 
        : capacity(initial_capacity), nnz_count(0) {
        static_assert(sizeof...(dims) == Order, "Number of dimensions must match Order");
        dimensions = {static_cast<size_t>(dims)...};
        
        // Allocate arrays for indices and values
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
        }
        values = new T[capacity];
    }

    // Constructor with dimensions array
    COOTensor(size_t initial_capacity, const std::array<size_t, Order>& dims) 
        : dimensions(dims), capacity(initial_capacity), nnz_count(0) {
        // Allocate arrays for indices and values
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
        }
        values = new T[capacity];
    }

    // Destructor
    ~COOTensor() {
        for (size_t i = 0; i < Order; ++i) {
            delete[] indices[i];
        }
        delete[] values;
    }

    // Copy constructor
    COOTensor(const COOTensor& other) 
        : dimensions(other.dimensions), capacity(other.capacity), nnz_count(other.nnz_count) {
        
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
            std::memcpy(indices[i], other.indices[i], nnz_count * sizeof(size_t));
        }
        values = new T[capacity];
        std::memcpy(values, other.values, nnz_count * sizeof(T));
    }

    // Assignment operator
    COOTensor& operator=(const COOTensor& other) {
        if (this != &other) {
            // Free existing resources
            for (size_t i = 0; i < Order; ++i) {
                delete[] indices[i];
            }
            delete[] values;
            
            // Copy new data
            dimensions = other.dimensions;
            capacity = other.capacity;
            nnz_count = other.nnz_count;
            
            for (size_t i = 0; i < Order; ++i) {
                indices[i] = new size_t[capacity];
                std::memcpy(indices[i], other.indices[i], nnz_count * sizeof(size_t));
            }
            values = new T[capacity];
            std::memcpy(values, other.values, nnz_count * sizeof(T));
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const COOTensor& obj) {
        os << "Tensor info: shape = (";
        for (size_t i = 0; i < Order; ++i) {
            os << obj.dimensions[i];
            if (i < Order - 1) {
                os << ",";
            }
        }
        os << "), nnz = " << obj.nnz_count;
        os << ", density = " <<  obj.get_density() << ".";
        return os;
    }

    // Reset the tensor by changing index, value, ...
    template<typename... Dims>
    void reset(size_t initial_capacity, Dims... dims) {
        capacity = initial_capacity;
        nnz_count = 0;
        
        static_assert(sizeof...(dims) == Order, "Number of dimensions must match Order");
        dimensions = {static_cast<size_t>(dims)...};
    
        // Allocate arrays for indices and values
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
        }
        values = new T[capacity];
    }

    // Resize arrays when capacity is reached
    void resize(size_t new_capacity) {
        for (size_t dim = 0; dim < Order; ++dim) {
            size_t* new_indices = new size_t[new_capacity];
            std::memcpy(new_indices, indices[dim], nnz_count * sizeof(size_t));
            delete[] indices[dim];
            indices[dim] = new_indices;
        }
        
        T* new_values = new T[new_capacity];
        std::memcpy(new_values, values, nnz_count * sizeof(T));
        delete[] values;
        values = new_values;
        
        capacity = new_capacity;
    }

    // Add a non-zero element to the tensor
    template<typename... Idx>
    void add_element(T value, Idx... idx) {
        static_assert(sizeof...(idx) == Order, "Number of indices must match Order");
        
        std::array<size_t, Order> idx_array = {static_cast<size_t>(idx)...};
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        if (value != T(0)) {  // Only store non-zero values
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            
            for (size_t i = 0; i < Order; ++i) {
                indices[i][nnz_count] = idx_array[i];
            }
            values[nnz_count] = value;
            nnz_count++;
        }
    }

    // Helper function to add element using array of indices
    void add_element_array(T value, const std::array<size_t, Order>& idx_array) {
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        if (value != T(0)) {
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            
            for (size_t i = 0; i < Order; ++i) {
                indices[i][nnz_count] = idx_array[i];
            }
            values[nnz_count] = value;
            nnz_count++;
        }
    }

    void update_element(T value, const std::array<size_t, Order>& idx_array) {
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        for (size_t n = 0; n < nnz_count; ++n) {
            bool flag = true;
            for (size_t i = 0; i < Order; ++i) {
                if (indices[i][n] != idx_array[i]) {
                    flag = false;
                }
            }
            if (flag) {
                values[n] += value;
                return;
            }
        }

        add_element_array(value, idx_array);
    }

    // Get the value at a specific position
    template<typename... Idx>
    T get(Idx... idx) const {
        static_assert(sizeof...(idx) == Order, "Number of indices must match Order");
        
        std::array<size_t, Order> idx_array = {static_cast<size_t>(idx)...};
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        for (size_t i = 0; i < nnz_count; ++i) {
            bool match = true;
            for (size_t dim = 0; dim < Order; ++dim) {
                if (indices[dim][i] != idx_array[dim]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return values[i];
            }
        }
        return T(0);
    }

    // Get number of non-zero elements
    size_t nnz() const {
        return nnz_count;
    }

    size_t get_capacity() const {
        return capacity;
    }

    std::array<size_t, Order> get_dimensions() const {
        return dimensions;
    }

    std::array<size_t*, Order> get_indices() const {
        return indices;
    }

    T* get_values() const {
        return values;
    }    

    double get_density() const {
        size_t N = 1;
        for (size_t dim = 0; dim < Order; ++dim) {
            N *= dimensions[dim];
        }
        double density = double(nnz_count) / double(N);
        return density;
    }

    // Sort elements lexicographically by indices
    void sort() {
        size_t* idx = new size_t[nnz_count];
        for (size_t i = 0; i < nnz_count; ++i) {
            idx[i] = i;
        }

        std::sort(idx, idx + nnz_count,
            [this](size_t i1, size_t i2) {
                return this->compare_indices(i1, i2);
            });

        // Create temporary arrays for sorting
        std::array<size_t*, Order> new_indices;
        for (size_t dim = 0; dim < Order; ++dim) {
            new_indices[dim] = new size_t[capacity];
            for (size_t i = 0; i < nnz_count; ++i) {
                new_indices[dim][i] = indices[dim][idx[i]];
            }
        }

        T* new_values = new T[capacity];
        for (size_t i = 0; i < nnz_count; ++i) {
            new_values[i] = values[idx[i]];
        }

        // Swap pointers
        for (size_t dim = 0; dim < Order; ++dim) {
            delete[] indices[dim];
            indices[dim] = new_indices[dim];
        }
        delete[] values;
        values = new_values;
        delete[] idx;
    }

    // Print the tensor in COO format
    void print() const {
        std::cout << "COO Tensor (";
        for (size_t i = 0; i < Order; ++i) {
            std::cout << dimensions[i];
            if (i < Order - 1) std::cout << " x ";
        }
        std::cout << "), " << nnz_count << " non-zero elements:\n";

        for (size_t i = 0; i < nnz_count; ++i) {
            std::cout << "(";
            for (size_t dim = 0; dim < Order; ++dim) {
                std::cout << indices[dim][i];
                if (dim < Order - 1) std::cout << ", ";
            }
            std::cout << ") = " << values[i] << "\n";
        }
    }

    tblis::tensor<T> to_dense() const {
        size_t N = 1;
        for (size_t i = 0; i < Order; ++i) {
            N *= dimensions[i];
        }
        tblis::tensor<T> fullT(dimensions);
        std::fill(fullT.data(), fullT.data() + N, 0.0);
        size_t idt;
        size_t ddt;
        for (size_t i = 0; i < nnz_count; ++i) {
            idt = 0;
            for (size_t dim = 0; dim < Order; ++dim) {
                ddt = 1;
                for (size_t od = dim + 1; od < Order; ++od) {
                    ddt *= dimensions[od];
                }
                idt += indices[dim][i] * ddt;
            }
            fullT.data()[idt] = values[i];
        }
        return fullT;
    }

    // Reshape the tensor into a 2d matrix
    COOMatrix_l2<T> reshape2Matl2(size_t mat_row, size_t mat_col) const {
        size_t N = 1;              // Get the total size: n1 * n2 * ... * nd 
        for (size_t i = 0; i < Order; ++i)
            N *= dimensions[i];
        if (mat_row * mat_col != N) {
            throw std::runtime_error("Input size does not match with the tensor size!");
        }

        // Reshape
        COOMatrix_l2<T> M(mat_row, mat_col, nnz_count);
        M.nnz_count = nnz_count;
        std::memcpy(M.values, values, nnz_count * sizeof(T));
        size_t idt;
        size_t ddt;
        for (size_t i = 0; i < nnz_count; ++i) {
            idt = 0;
            for (size_t dim = 0; dim < Order; ++dim) {
                ddt = 1;
                for (size_t od = dim + 1; od < Order; ++od) {
                    ddt *= dimensions[od];
                }
                idt += indices[dim][i] * ddt;
            }
            //fullT.data()[idt] = values[i];
            M.row_indices[i] = idt / mat_col;
            M.col_indices[i] = idt % mat_col;
        }

        return M;
    }

    // Simplified contraction function for single dimension
    template<size_t OtherOrder>
    COOTensor<T, Order + OtherOrder - 2> contract(
        const COOTensor<T, OtherOrder>& other,
        size_t this_dim,    // Dimension to contract from this tensor
        size_t other_dim    // Dimension to contract from other tensor
    ) const {
        if (this_dim >= Order || other_dim >= OtherOrder) {
            throw std::out_of_range("Contraction dimensions out of bounds");
        }
        
        // Calculate output dimensions
        constexpr size_t ResultOrder = Order + OtherOrder - 2;
        std::array<size_t, ResultOrder> out_dims;
        size_t out_idx = 0;
        
        // Map dimensions to output tensor
        for (size_t i = 0; i < Order; ++i) {
            if (i != this_dim) {
                out_dims[out_idx++] = dimensions[i];
            }
        }
        for (size_t i = 0; i < OtherOrder; ++i) {
            if (i != other_dim) {
                out_dims[out_idx++] = other.get_dimensions()[i];
            }
        }

        // Create result tensor
        size_t initial_capacity = std::min(nnz_count * other.nnz(), 
                                         capacity + other.get_capacity());
        COOTensor<T, ResultOrder> result(initial_capacity, out_dims);
        
        // For each non-zero element in this tensor
        for (size_t i = 0; i < nnz_count; ++i) {
            // For each non-zero element in other tensor
            for (size_t j = 0; j < other.nnz(); ++j) {
                // Check if contracted indices match
                if (indices[this_dim][i] == other.get_indices()[other_dim][j]) {
                    T prod = values[i] * other.get_values()[j];
                    //if (prod != T(0)) {
                    if (std::abs(prod) > 1e-14) {
                        // Build output indices
                        std::array<size_t, ResultOrder> out_indices;
                        out_idx = 0;
                        
                        // Map indices from this tensor
                        for (size_t k = 0; k < Order; ++k) {
                            if (k != this_dim) {
                                out_indices[out_idx++] = indices[k][i];
                            }
                        }
                        
                        // Map indices from other tensor
                        for (size_t k = 0; k < OtherOrder; ++k) {
                            if (k != other_dim) {
                                out_indices[out_idx++] = other.get_indices()[k][j];
                            }
                        }

                        //result.add_element_array(prod, out_indices);

                        result.update_element(prod, out_indices);
                    }
                }
            }
        }
        
        return result;
    }

    // Write data to file
    void write_to_file(const std::string& filename) const {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        // Write first line: dimensions and data type
        outfile << "Order: ";
        for (size_t i = 0; i < Order; ++i) {
            outfile << dimensions[i];
            if (i < Order - 1) outfile << " ";
        }
        outfile << ", NNZ: " << nnz_count;
        outfile << ", Datatype: " << typeid(T).name() << "\n";

        // Write each non-zero element
        for (size_t i = 0; i < nnz_count; ++i) {
            // Write indices
            for (size_t dim = 0; dim < Order; ++dim) {
                outfile << indices[dim][i];
                if (dim < Order - 1) outfile << " ";
            }
            // Write value
            outfile << " " << std::fixed << std::setprecision(6) << values[i] << "\n";
        }

        outfile.close();
    }

    // Read data
    void read_from_file(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }

        std::string line;
        // Skip the first line (dimensions and type)
        std::getline(infile, line);
        
        // Reset the number of non-zeros
        nnz_count = 0;
        
        // Read data lines
        std::array<size_t, Order> curr_indices;
        T value;
        
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            
            // Read indices and value
            for (size_t i = 0; i < Order; ++i) {
                iss >> curr_indices[i];
            }
            iss >> value;
            
            // Add element to tensor
            add_element_array(value, curr_indices);
        }
        
        infile.close();
    }

    // Random data generator
    void generate_random(double density, 
                        Distribution dist = Distribution::UNIFORM,
                        const DistributionParams& params = DistributionParams(),
                        unsigned seed = std::random_device{}()) {
        if (density < 0.0 || density > 1.0) {
           throw std::invalid_argument("Density must be between 0 and 1");
        }

        // Reset nnz count
        nnz_count = 0;
        
        // Random number generators
        //std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> density_dist(0.0, 1.0);
        
        // Create the selected distribution
        auto value_dist = [&gen, &dist, &params]() -> T {
            switch (dist) {
                case Distribution::UNIFORM: {
                    std::uniform_real_distribution<> d(params.min_value, params.max_value);
                    return static_cast<T>(d(gen));
                }
                case Distribution::NORMAL: {
                    std::normal_distribution<> d(params.mean, params.std_dev);
                    return static_cast<T>(d(gen));
                }
                case Distribution::STANDARD_NORMAL: {
                    std::normal_distribution<> d(0.0, 1.0);
                    return static_cast<T>(d(gen));
                }
                case Distribution::GAMMA: {
                    std::gamma_distribution<> d(params.gamma_shape, params.gamma_scale);
                    return static_cast<T>(d(gen));
                }
                default:
                    throw std::invalid_argument("Unknown distribution type");
            }
        };

        // Array to store current indices
        std::array<size_t, Order> curr_indices;
        std::fill(curr_indices.begin(), curr_indices.end(), 0);
        
        // Iterate through all possible positions
        bool done = false;
        while (!done) {
            // Check if this position should be non-zero
            if (density_dist(gen) < density) {
                // Generate random value using selected distribution
                T value = value_dist();
                
                // Add element (will handle capacity resize if needed)
                add_element_array(value, curr_indices);
            }

            // Increment indices
            for (int dim = Order - 1; dim >= 0; --dim) {
                curr_indices[dim]++;
                if (curr_indices[dim] < dimensions[dim]) {
                    break;
                }
                curr_indices[dim] = 0;
                if (dim == 0) {
                    done = true;
                }
            }
        }

        // Sort the tensor (Needed?)
        sort();
    }

    // Binary addition operator
    COOTensor operator+(const COOTensor& other) const {
        // Check if dimensions match
        if (dimensions != other.dimensions) {
            throw std::invalid_argument("Tensor dimensions must match for addition");
        }

        // Create result tensor with initial capacity
        size_t initial_capacity = nnz_count + other.nnz_count;
        COOTensor result(initial_capacity, dimensions);

        // Create temporary arrays for all possible elements
        std::vector<std::array<size_t, Order>> all_indices;
        std::vector<T> all_values;

        // First, add all elements from the first tensor
        for (size_t i = 0; i < nnz_count; ++i) {
            std::array<size_t, Order> curr_indices;
            for (size_t dim = 0; dim < Order; ++dim) {
                curr_indices[dim] = indices[dim][i];
            }
            all_indices.push_back(curr_indices);
            all_values.push_back(values[i]);
        }

        // Then process elements from the second tensor
        for (size_t i = 0; i < other.nnz_count; ++i) {
            std::array<size_t, Order> curr_indices;
            for (size_t dim = 0; dim < Order; ++dim) {
                curr_indices[dim] = other.indices[dim][i];
            }

            // Look for matching indices in our temporary arrays
            bool found = false;
            for (size_t j = 0; j < all_indices.size(); ++j) {
                bool match = true;
                for (size_t dim = 0; dim < Order; ++dim) {
                    if (all_indices[j][dim] != curr_indices[dim]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    // Update existing value
                    all_values[j] += other.values[i];
                    found = true;
                    break;
                }
            }

            // If no matching indices found, add new element
            if (!found) {
                all_indices.push_back(curr_indices);
                all_values.push_back(other.values[i]);
            }
        }

        // Add all non-zero values to the result tensor
        for (size_t i = 0; i < all_indices.size(); ++i) {
            if (all_values[i] != T(0)) {
                result.add_element_array(all_values[i], all_indices[i]);
            }
        }

        // Sort the result tensor
        result.sort();
        return result;
    }

    // Addition assignment operator
    COOTensor& operator+=(const COOTensor& other) {
        *this = *this + other;
        return *this;
    }

    double rel_diff(const COOTensor& other) {
        // Check if dimensions match
        if (dimensions != other.dimensions) {
            throw std::invalid_argument("Tensor dimensions must match for difference comparison!");
        }
        double diff = 0.0;
        double denom = 0.0;
        for (size_t i = 0; i < nnz_count; ++i) {
            T val_this = values[i];
            T val_that;
            T temp;
            bool flag = false;
            for (size_t j = 0; j < other.nnz_count; ++j) {
                for (size_t d = 0; d < Order; ++d) {                    
                    if (other.indices[d][j] != indices[d][i]) {
                        break;
                    }   
                    if (d == Order - 1) {
                        flag = true;
                        val_that = other.values[j];
                        temp = std::abs(val_that - val_this);
                        diff += temp * temp;
                    }  
                }
                if (flag == true) {
                    break;
                }
            }
            if (flag == false) {
                diff += val_this * val_this;        
            }
            denom += val_this * val_this;
        }
        return std::sqrt(diff / denom);
    }
};

// Contract function for tensor train contraction
template<typename T, size_t Order1, size_t Order2>
auto contract_tt_cores(const COOTensor<T, Order1>& result, 
                      const COOTensor<T, Order2>& next_core,
                      size_t contraction_dim) {
    return result.contract(next_core, contraction_dim, 0);
}

// Base case for single tensor
template<typename T, size_t Order>
auto SparseTTtoTensor(const COOTensor<T, Order>& tensor) {
    return tensor;
}

// Recursive case for multiple tensors
template<typename T, size_t Order1, size_t Order2, typename... Rest>
auto SparseTTtoTensor(const COOTensor<T, Order1>& first,
                      const COOTensor<T, Order2>& second,
                      const Rest&... rest) {  
    // Contract first two tensors
    auto intermediate = first.contract(second, Order1 - 1, 0);
    
    // Recursively contract the result with remaining tensors
    if constexpr (sizeof...(rest) > 0) {
        return SparseTTtoTensor(intermediate, rest...);
    } else {
        return intermediate;
    }
}

// Function to verify tensor train structure
template<typename T, typename... Tensors>
void verify_tt_structure(const Tensors&... tensors) {
    std::vector<size_t> orders = {Tensors::Order...};
    
    // Check minimum number of cores
    if (sizeof...(Tensors) < 2) {
        throw std::invalid_argument("Tensor train must have at least 2 cores");
    }
    
    // Check first and last cores are order 2
    if (orders.front() != 2 || orders.back() != 2) {
        throw std::invalid_argument("First and last cores must be order 2");
    }
    
    // Check middle cores are order 3
    for (size_t i = 1; i < orders.size() - 1; ++i) {
        if (orders[i] != 3) {
            throw std::invalid_argument("Middle cores must be order 3");
        }
    }
}

// Tensor-train contraction main function
template<typename T, typename... Tensors>
auto SparseTTtoTensor(const Tensors&... tensors) {
    // Verify tensor train structure
    verify_tt_structure<T>(tensors...);
    // Perform contraction
    return SparseTTtoTensor<T>(tensors...);
}

// TT Result struct
struct SparseTTRes {
    COOTensor<double, 2> StartG;
    COOTensor<double, 2> EndG;
    std::vector<COOTensor<double, 3>> InterG;
};

// Declare the template function of TT-ID
template<typename T, size_t Order> SparseTTRes 
TT_ID_sparse(const COOTensor<T, Order>& tensor, double const cutoff, size_t const r_max,
             double const spthres, bool check_flag, bool cross_flag);
              
// Explicitly declare the specializations you'll use
extern template SparseTTRes TT_ID_sparse<double, 3>(const COOTensor<double, 3>&, double, size_t, double, bool, bool);
extern template SparseTTRes TT_ID_sparse<double, 4>(const COOTensor<double, 4>&, double, size_t, double, bool, bool);
extern template SparseTTRes TT_ID_sparse<double, 5>(const COOTensor<double, 5>&, double, size_t, double, bool, bool);
extern template SparseTTRes TT_ID_sparse<double, 6>(const COOTensor<double, 6>&, double, size_t, double, bool, bool);
// Add other specializations as needed

#endif