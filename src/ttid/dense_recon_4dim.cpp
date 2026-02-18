#include "dtensor.h"
#include "densett.h"
#include "sptensor.h"
#include "util.h"

int main()
{
    int dim = 4;
    std::array<size_t, 4> mode = {100, 100, 100, 100};
    std::array<size_t, 3> rank = {100, 2000, 100};
    
    std::vector<std::array<size_t, 3>> dim_list;
    dim_list.push_back({1, mode[0], rank[0]});
    dim_list.push_back({rank[0], mode[1], rank[1]});
    dim_list.push_back({rank[1], mode[2], rank[2]});
    dim_list.push_back({rank[2], mode[3], 1});

    std::vector<tblis::tensor<double>> ttList;
  
    for (int i = 0; i < dim; ++i) {
        std::array<size_t, 3> dimensions = dim_list[i];
        tblis::tensor<double> TTcore(dimensions);

        size_t N = 1;
        for (size_t i = 0; i < 3; ++i) {
            N *= dimensions[i];
        }

        std::fill(TTcore.data(), TTcore.data() + N, 2 * i + 3);
        ttList.push_back(TTcore);
    }

    {
    util::Timer timer("Reconstruction evaluation");
    tblis::tensor<double> result_with_trivial;
    auto recTensor = denseT::TT_Contraction_dense(ttList);
    }

    util::Timer::summarize();
    return 0;
}

