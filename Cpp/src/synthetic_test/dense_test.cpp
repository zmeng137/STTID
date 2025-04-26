#include "dtensor.h"
#include "densett.h"
#include "sptensor.h"
#include "util.h"

enum dense_tt_id {
    TTSVD = 1,
    TTID_PRRLDU = 2,
    TTID_RRQR = 3
};

void DenseSyntheticT_DenseTest(std::initializer_list<int> tShape, std::initializer_list<int> tRank, 
                               dense_tt_id ttalgoType)
{
    std::cout << "Dense TT test of a synthetic dense tensor starts.\n"; 
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    // Generate a synthetic dense tensor {tShape} (TT rank = {tRank})
    auto synTensor = denseT::SyntheticTenGen<double>(tShape, tRank);

    // Tensor train decomposition
    int maxdim = std::max(tRank);
    double cutoff = 1E-10; 
    std::vector<tblis::tensor<double>> ttList;
    switch (ttalgoType) {
    case TTSVD:
        std::cout << "TT-SVD starts\n";
        start = std::chrono::high_resolution_clock::now();
        ttList = TT_SVD_dense(synTensor, maxdim, cutoff);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "TT-SVD ends. It took " << elapsed.count() << " seconds.\n";
        break;
    case TTID_PRRLDU:
        std::cout << "TT-ID-PRRLDU starts\n";
        start = std::chrono::high_resolution_clock::now();
        ttList = TT_IDPRRLDU_dense(synTensor, maxdim, cutoff);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "TT-ID-PRRLDU ends. It took " << elapsed.count() << " seconds.\n";
        break;
    case TTID_RRQR:
        std::cout << "...TDID_RRQR is to be debugged...\n";
        break;
    default:
        std::cout << "Please give a correct no. of TT algorithm types.\n";
        break;
    }   

    // Reconstruction evaluation
    auto recTensor = denseT::TT_Contraction_dense(ttList);
    double error = denseT::NormError(synTensor, recTensor, 2, true);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Test ends." << std::endl;
}

void SparseSyntheticT_DenseTest(std::initializer_list<int> tShape, std::initializer_list<int> tRank, 
                                std::initializer_list<double> tDensity, dense_tt_id ttalgoType)
{
    std::cout << "Dense TT test of a synthetic sparse tensor starts.\n"; 
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    // Generate a synthetic sparse tensor {tShape} (TT rank = {tRank})
    auto synTensor = denseT::SyntheticSparseTenGen<double>(tShape, tRank, tDensity);
    int nnz = 0;
    int nbar = denseT::GetSize(synTensor);
    for (int i = 0; i < nbar; i++) {
        if (std::abs(synTensor.data()[i]) > 1e-10)
            nnz += 1;
    }
    std::cout << "Synthetic tensor info: nnz = " << nnz << ", density = " << double(nnz) / double(nbar) << "\n";

    // Tensor train decomposition
    int maxdim = std::max(tRank);
    double cutoff = 1E-10; 
    std::vector<tblis::tensor<double>> ttList;
    switch (ttalgoType) {
    case TTSVD:
        std::cout << "TT-SVD starts\n";
        start = std::chrono::high_resolution_clock::now();
        ttList = TT_SVD_dense(synTensor, maxdim, cutoff);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "TT-SVD ends. It took " << elapsed.count() << " seconds.\n";
        break;
    case TTID_PRRLDU:
        std::cout << "TT-ID-PRRLDU starts\n";
        start = std::chrono::high_resolution_clock::now();
        ttList = TT_IDPRRLDU_dense(synTensor, maxdim, cutoff);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "TT-ID-PRRLDU ends. It took " << elapsed.count() << " seconds.\n";
        break;
    case TTID_RRQR:
        std::cout << "...TDID_RRQR is to be debugged...\n";
        break;
    default:
        std::cout << "Please give a correct no. of TT algorithm types.\n";
        break;
    }   

    // Reconstruction evaluation
    std::cout << "Tensor factor info:\n";
    for (int f = 0; f < tShape.size(); ++f) {
        int factor_nnz = 0;
        int factor_nbar = denseT::GetSize(ttList[f]);
        for (int i = 0; i < factor_nbar; ++i) {
        if (std::abs(ttList[f].data()[i]) > 1e-10)
            factor_nnz += 1;
        }
        std::cout << "factor " << f << ": nnz = " << factor_nnz << ", density = " << double(factor_nnz) / double(factor_nbar) << "\n";
    }
    auto recTensor = denseT::TT_Contraction_dense(ttList);
    double error = denseT::NormError(synTensor, recTensor, 2, true);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Test ends." << std::endl;
}

void toy_test()
{
    // Initialize a very simple tensor
    int d1 = 5, d2 = 10, d3 = 6, d4 = 7, d5 = 11;
    tblis::tensor<double> tensor({d1, d2, d3, d4, d5});
    for (int i = 0; i < d1; ++i)
        for (int j = 0; j < d2; ++j)
            for (int m = 0; m < d3; ++m)
                for (int n = 0; n < d4; ++n)
                    for (int l = 0; l < d5; ++l)
                        tensor(i, j, m, n) = i * j - m + n * l;
    //std::cout << "The toy tensor is \n" << tensor << std::endl;
    auto ttList = TT_IDPRRLDU_dense(tensor, 100, 1e-10);
    auto recTensor = denseT::TT_Contraction_dense(ttList);
    double error = denseT::NormError(tensor, recTensor, 2, false);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Synthetic test 1 ends." << std::endl;
}

int main(int argc, char** argv) 
{
    //{util::Timer timer("main process");
    //SparseSyntheticT_DenseTest({100, 100, 100, 100}, {40, 200, 50}, {1E-2, 5E-3, 6E-3, 1E-2}, TTID_PRRLDU);}
    //util::Timer::summarize();
    
    // Memory logger
    //util::HostMemoryLogger host_mem_logger("gpu_main_memory_usage.csv", 1000);
    //util::DeviceMemoryLogger device_mem_logger("gpu_device_memory_usage.csv", 1000, 0);
    
    {util::Timer timer("Main process");
    //------------ GLOBAL PARAMETERS ------------//
    // Input tensor settings
    const std::string filepath = "/home/zmeng5/TensorData/7.tns";
    //const std::string filepath = "/home/mengzn/Desktop/TensorData/20.tns";
    const size_t Order = 4;    
    const size_t num_entries = 156492; 
    const std::array<size_t, Order> dimensions = {100, 100, 100, 100};
    
    // Tensor-train algorithm settings
    int r_max = 1000;
    double eps = 1e-8;
    //-------------------------------------------//
    
    // Create arrays to store the data
    std::array<size_t*, Order> indices;
    double* values = new double[num_entries];
    bool data_lf;
    // Read the data
    {
        util::Timer timer("Data load");
        data_lf = util::read_sparse_tensor<Order>(filepath, indices, values, num_entries);
    }
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
    }
    
    }util::Timer::summarize();
    return 0; 
}