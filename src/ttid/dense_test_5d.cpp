#include "dtensor.h"
#include "densett.h"
#include "sptensor.h"
#include "util.h"

int main(int argc, char** argv) 
{
    if (argc != 12) {
        std::cerr << "Error: Expected 11 arguments, got " << (argc - 1) << std::endl;
        std::cerr << "Usage: " << argv[0] << " <data_file_path> <nnz> <order1> <order2> <order3> <order4> <order5> <rmax> <epsilon> <binary> <idx_offset>" << std::endl;
        return 1;
    } 

    //------------ GLOBAL PARAMETERS ------------//
    // Input tensor settings
    std::string filepath = argv[1];
    const size_t Order = 5;    
    size_t num_entries = std::stoll(argv[2]); 
    std::array<size_t, Order> dimensions;

    // Tensor-train algorithm settings
    size_t r_max = std::stoll(argv[8]);
    double eps = std::stod(argv[9]);
    bool binary = std::stoi(argv[10]);
    size_t idx_offset = std::stoll(argv[11]);
    
    // Flags
    bool check_flag = false;   
    //bool cross_flag = true;
    bool ifEval = false;

    // Print loading info
    std::cout << "Tensor file: " << filepath << "\n" << "Nonzero count: " << num_entries << "\n";
    std::cout << "r_max: " << r_max << "\n" << "tolerance: " << eps << "\n" << "spthres: NULL" << std::endl;
    //-------------------------------------------//

    // Create arrays to store the data
    std::array<size_t*, Order> indices;
    double* values = new double[num_entries];
    bool data_lf;
    
    {util::Timer timer("Data load");  // Read the data
    data_lf = util::read_sparse_tensor<Order>(filepath, indices, dimensions, values, num_entries, binary, idx_offset);}

    // Adaptive dimensions
    std::cout << "Dimension of the input tensor: ";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        size_t temp = std::stoll(argv[3 + i]);
        dimensions[i] = std::max(dimensions[i], temp);
        std::cout << dimensions[i] << ", ";
    }
    std::cout << "\n";

    // Print the input data
    if (data_lf) {
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

        // Construct tensor
        COOTensor<double, Order> Tensor;
        Tensor.dimensions = dimensions;
        Tensor.capacity = num_entries;
        Tensor.nnz_count = num_entries;
        Tensor.indices = indices;
        Tensor.values = values;
        
        auto dTensor  = Tensor.to_dense();

        // Sparse TTID algorithm
        std::cout << "TT-ID-PRRLDU starts\n";
        auto ttList = TT_IDPRRLDU_dense(dTensor, r_max, eps);        
        
        // Output information display
        if (ifEval) {
            util::Timer timer("Result evaluation");
            tblis::tensor<double> result_with_trivial;
            auto recTensor = denseT::TT_Contraction_dense(ttList);
            //double error = denseT::NormError(dTensor, recTensor, 2, true);
            //std::cout << "TT recon error: " << error << "\n";
            //std::cout << "Test ends." << std::endl;
        }
        
        // Timer summary
        util::Timer::summarize();
    }
    return 0;
}

