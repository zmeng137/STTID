// Sparse matrix/tensor   
#include "sptensor.h"
#include "spmatrix.h"

// Functions/Result sets
#include "spfunctions.h"
#include "structures.h"

// Tools
#include "util.h"
#include "cutil.h"

int main(int argc, char* argv[]) 
{ 
    if (argc != 13) {
        std::cerr << "Error: Expected 12 arguments, got " << (argc - 1) << std::endl;
        std::cerr << "Usage: " << argv[0] << " <data_file_path> <nnz> <order1> <order2> <order3> <order4> <order5> <rmax> <epsilon> <spthres> <binary> <idx_offset>" << std::endl;
        return 1;
    }  
    
    //------------ GLOBAL PARAMETERS ------------//
    // Input tensor settings
    std::string filepath = argv[1];
    const size_t Order = 4;    
    size_t num_entries = std::stoll(argv[2]); 
    std::array<size_t, Order> dimensions;
    dimensions[0] = std::stoll(argv[3]);
    dimensions[1] = std::stoll(argv[4]);
    dimensions[2] = std::stoll(argv[5]);
    dimensions[3] = std::stoll(argv[6]);
    dimensions[4] = std::stoll(argv[7]);

    // Tensor-train algorithm settings
    size_t r_max = std::stoll(argv[8]);
    double eps = std::stod(argv[9]);
    double spthres = std::stod(argv[10]);
    bool binary = std::stoi(argv[11]);
    size_t idx_offset = std::stoll(argv[12]);
    bool verbose = false;  
    bool ifEval = false;  

    // Print loading info
    std::cout << "Tensor file: " << filepath << "\n" << "Nonzero count: " << num_entries << "\n";
    std::cout << "r_max: " << r_max << "\n" << "tolerance: " << eps << "\n" << "spthres: " << spthres << std::endl;
    for (size_t i = 0; i < dimensions.size(); ++i) std::cout << "Dimension input by argument [" << i << "]: " << dimensions[i] << "\n";
    //-------------------------------------------//

    // Create arrays to store the data
    std::array<size_t*, Order> indices;
    double* values = new double[num_entries];
    bool data_lf;
    
    {util::Timer timer("Data load");  // Read the data
    data_lf = util::read_sparse_tensor<Order>(filepath, indices, dimensions, values, num_entries, binary, idx_offset);}
    
    if (data_lf) {
        // Print the input data
        std::cout << "The input tensor in COO format is as follows:\n";
        for (int i = 0; i < Order; ++i) {
            std::cout << "index " << i << ": "; 
            for (int j = 0; j < (num_entries > 5 ? 5 : num_entries); ++j) {
                std::cout << indices[i][j] << "  ";
            }
            std::cout << "...\n";
        }
        std::cout << "values: ";
        for (int j = 0; j < (num_entries > 5 ? 5 : num_entries); ++j) {
           std::cout << values[j] << "  ";     
        }
        std::cout << "...\n";

        for (size_t i = 0; i < dimensions.size(); ++i) std::cout << "Dimension after data loading [" << i << "]: " << dimensions[i] << "\n";

        // Construct tensor
        COOTensor<double, Order> Tensor;
        Tensor.dimensions = dimensions;
        Tensor.capacity = num_entries;
        Tensor.nnz_count = num_entries;
        Tensor.indices = indices;
        Tensor.values = values;
        
        // Sparse TTID algorithm
        // Memory logger
        //util::HostMemoryLogger host_mem_logger("gpu_main_memory_usage.csv", 100);
        //util::DeviceMemoryLogger device_mem_logger("gpu_device_memory_usage.csv", 1000, 0);
        util::getCuversion();  // Get CUDA info
        std::cout << "SPARSE TT-ID STARTS:\n";
        auto ttList = TT_ID_sparse(Tensor, eps, spthres, r_max, verbose);
        
        // Output information display
        if (ifEval) {
            // Sparse information
            std::cout << "OUTPUT INFO:\n"; 
            std::cout << "Output core F1 --" << ttList.StartG << "\n";
            std::cout << "Output core F2 --" << ttList.InterG[0] << "\n";
            std::cout << "Output core F3 --" << ttList.InterG[1] << "\n";
            std::cout << "Output core F3 --" << ttList.InterG[2] << "\n";
            std::cout << "Output core F4 --" << ttList.EndG << "\n";
            util::Timer timer("Result evaluation");
            auto reconT = SparseTTtoTensor<double>(ttList.StartG, ttList.InterG[0], ttList.InterG[1], ttList.EndG);
            std::cout << "Reconstructed tensor -- " << reconT << "\n";
            double err = Tensor.rel_diff(reconT);
            std::cout << "Relative reconstruction error = " << err << std::endl;
        }
        
        // Timer summary
        util::Timer::summarize();
    }
    return 0; 
}