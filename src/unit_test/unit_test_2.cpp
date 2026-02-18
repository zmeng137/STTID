#include <gtest/gtest.h>
#include <tblis/tblis.h>
#include "core.h"

TEST(TblisTEST, tensor_contraction_1)
{   
    tblis::tensor<double> A({2, 4, 3});
    tblis::tensor<double> B({3, 4, 5});
    tblis::tensor<double> C({2, 5});
    
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                A(i,j,k) = i + j + k;
    
    // B initialization
    for (int i=0; i<3; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<5; ++k)
                B(i,j,k) = i + j - k;

    // C <- contraction of A and B 
    tblis::mult<double>(1.0, A, "abc", B, "cbd", 0.0, C, "ad");

    int answer[2][5] = {{98, 68, 38, 8, -22}, 
                        {128, 86, 44, 2, -40}};
    
    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<5; ++j) 
            max_error = std::max(answer[i][j] - C(i,j), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}

TEST(TblisTEST, tensor_permute_1)
{
    tblis::tensor<float> A({2, 4, 3});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                A(i,j,k) = i + j + k;

    auto B = A.permuted({2,1,0});
    // TO BE TESTED...
}

TEST(TblisTEST, tensor_reshape_1)
{   
    // Reshape A(2,4,3)->B(12,2)->C(2,4,3)
    tblis::tensor<float> A({2, 4, 3});
    tblis::tensor<float> B({12, 2});
    tblis::tensor<float> C({2, 4, 3});

    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                A(i,j,k) = i + j + k;
    
    auto dataA = A.data();
    auto dataB = B.data();
    auto dataC = C.data();
    std::copy(dataA, dataA+2*4*3, dataB);
    std::copy(dataB, dataB+12*2, dataC);

    // Find the maximum error between A and C
    float max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k) 
                max_error = std::max(A(i,j,k) - C(i,j,k), max_error);                 
    EXPECT_NEAR(max_error,0,1E-7);
}