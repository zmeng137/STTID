#include <gtest/gtest.h>
#include "sptensor.h"
#include "spmatrix.h"

TEST(SparseMat_TEST, COO_MatMultiply_1)
{
    // Create two sparse matrices
    COOMatrix_l2<double> A(3, 3, 5);
    COOMatrix_l2<double> B(3, 3, 5);

    // Add some elements
    A.add_element(0, 0, 1.0);
    A.add_element(0, 2, 2.0);
    B.add_element(2, 1, 3.0);

    // Multiply
    COOMatrix_l2<double> C = A.multiply(B);
    //C.print();
}

TEST(SparseMat_TEST, COO_MatMultiply_2)
{
    // Create a 100x100 matrix
    COOMatrix_l2<double> A(6, 4, 24);
    COOMatrix_l2<double> B(4, 5, 20);

    // Generate random entries
    double density = 0.3;   // 50% non-zero elements
    unsigned int seed = 42; // Random seed for reproducibility
    A.generate_random(density, seed, -10.0, 10.0);

    seed = 76;
    B.generate_random(density, seed, -10.0, 10.0);

    COOMatrix_l2<double> C = A.multiply(B);
    //C.print();
}

TEST(SparseMat_TEST, COO_MatStruct)
{
    // Create a 4x4 sparse matrix
    COOMatrix_l2<double> matrix(4, 4);

    // Add some non-zero elements
    matrix.add_element(0, 0, 1.0);
    matrix.add_element(1, 1, 2.0);
    matrix.add_element(2, 3, 3.0);
    matrix.add_element(3, 2, 4.0);

    // Sort the elements
    matrix.sort();

    // Print the matrix
    //matrix.print();

    // Get a specific element
    //std::cout << "Value at (2,3): " << matrix.get(2, 3) << "\n";
    //std::cout << "Value at (1,2): " << matrix.get(1, 2) << "\n";

    // Test copy constructor
    COOMatrix_l2<double> matrix2 = matrix;
    
    //std::cout << "\nCopied matrix:\n";
    //matrix2.print();

    matrix2.addUpdate(0, 1, 3.4);    
    //matrix2.print();

    double* full = matrix2.todense();
    //util::PrintMatWindow(full, 4, 4, {0,3}, {0,3});
    delete[] full;
}

TEST(SparseMat_TEST, COO_MatReshape)
{
    // Create a 4x4 sparse matrix
    COOMatrix_l2<double> matrix(4, 5);

    // Add some non-zero elements
    matrix.add_element(0, 0, 1.0);
    matrix.add_element(1, 1, 3.0);
    matrix.add_element(2, 4, 5.0);
    matrix.add_element(3, 2, 2.0);

    matrix.reshape(2, 10);
    
    EXPECT_NEAR(1.0, matrix.get(0,0), 1E-10);
    EXPECT_NEAR(3.0, matrix.get(0,6), 1E-10);
    EXPECT_NEAR(5.0, matrix.get(1,4), 1E-10);
    EXPECT_NEAR(2.0, matrix.get(1,7), 1E-10);
}


TEST(SparseTensor_TEST, COO_TensorStruct)
{
// Create tensors of different orders
    COOTensor<double, 3> tensor3d(10, 3, 3, 3);  // 3rd order tensor
    COOTensor<double, 4> tensor4d(10, 2, 2, 2, 2);  // 4th order tensor
    COOTensor<double, 5> tensor5d(10, 2, 2, 2, 2, 2);  // 5th order tensor

    // Add elements to 3D tensor
    tensor3d.add_element(1.0, 0, 0, 0);
    tensor3d.add_element(2.0, 1, 1, 1);
    //tensor3d.print();

    // Add elements to 4D tensor
    tensor4d.add_element(1.0, 0, 0, 0, 0);
    tensor4d.add_element(2.0, 1, 1, 1, 1);
    //tensor4d.print();

    // Add elements to 5D tensor
    tensor5d.add_element(1.0, 0, 0, 0, 0, 0);
    tensor5d.add_element(2.0, 1, 1, 1, 1, 1);
    //tensor5d.print();
}

TEST(SparseTensor_TEST, COO_Sparse2Dense)
{
    // Create tensors of different orders
    COOTensor<double, 3> tensor3d(10, 3, 4, 5);       // 3rd order tensor
    COOTensor<double, 5> tensor5d(10, 3, 2, 4, 3, 1); // 5th order tensor
   
    tensor3d.add_element(1.0, 0, 0, 0);
    tensor3d.add_element(2.0, 1, 1, 1);
    tensor3d.add_element(3.0, 2, 3, 2);
    tensor3d.add_element(4.0, 0, 2, 3);
    auto tensor3d_full = tensor3d.to_dense(); 
    EXPECT_NEAR(1.0, tensor3d_full(0,0,0), 1E-10);
    EXPECT_NEAR(2.0, tensor3d_full(1,1,1), 1E-10);
    EXPECT_NEAR(3.0, tensor3d_full(2,3,2), 1E-10);
    EXPECT_NEAR(4.0, tensor3d_full(0,2,3), 1E-10);

    tensor5d.add_element(1.0, 0, 0, 0, 0, 0);
    tensor5d.add_element(2.0, 1, 1, 3, 1, 0);
    tensor5d.add_element(3.0, 2, 1, 2, 0, 0);
    tensor5d.add_element(4.0, 2, 0, 3, 2, 0);
    tensor5d.add_element(5.0, 1, 1, 1, 0, 0);
    auto tensor5d_full = tensor5d.to_dense(); 
    EXPECT_NEAR(1.0, tensor5d_full(0,0,0,0,0), 1E-10);
    EXPECT_NEAR(2.0, tensor5d_full(1,1,3,1,0), 1E-10);
    EXPECT_NEAR(3.0, tensor5d_full(2,1,2,0,0), 1E-10);
    EXPECT_NEAR(4.0, tensor5d_full(2,0,3,2,0), 1E-10);
    EXPECT_NEAR(5.0, tensor5d_full(1,1,1,0,0), 1E-10);
}

TEST(SparseTensor_TEST, COO_TensorContraction1)
{ 
    COOTensor<float, 2> A(10, 4, 6);  // 4x6 tensor
    COOTensor<float, 2> B(10, 6, 7);  // 6x7 tensor
    
    // Add some elements
    A.add_element(1.0f, 0, 1);
    A.add_element(2.0f, 1, 2);
    A.add_element(-2.5f,3, 3);
    B.add_element(3.0f, 2, 0);
    B.add_element(4.0f, 3, 1);

    // Contract A's second dimension (index 1) with B's first dimension (index 0)
    auto C = A.contract(B, 1, 0);  // Result will be 4x7 tensor
    
    // Check the correctness
    EXPECT_EQ(C.nnz(), 2);   // Number of non-zeros
    EXPECT_NEAR(6.0f, C.get(1,0), 1E-8);
    EXPECT_NEAR(-10.0f, C.get(3,1), 1E-8);
}

TEST(SparseTensor_TEST, COO_TensorContraction2)
{
    // Create tensors
    COOTensor<double, 3> A(100, 4, 5, 6);  // 4x5x6 tensor
    COOTensor<double, 3> B(100, 6, 7, 3);  // 6x7x3 tensor
    
    // Add some elements
    A.add_element(1.0, 0, 1, 2);
    A.add_element(2.0, 1, 2, 3);
    A.add_element(5.0, 3, 2, 5);
    A.add_element(-0.3,2, 4, 1);
    B.add_element(3.0, 2, 0, 1);
    B.add_element(4.0, 3, 1, 2);
    B.add_element(-1.2,5, 6, 2);
    
    // Contract A's third dimension (index 2) with B's first dimension (index 0)
    auto C = A.contract(B, 2, 0);  // Result will be 4x5x7 tensor
    
    // Check the correctness
    EXPECT_EQ(C.nnz(), 3);   // Number of non-zeros
    EXPECT_NEAR(3.0, C.get(0,1,0,1), 1E-10);
    EXPECT_NEAR(8.0, C.get(1,2,1,2), 1E-10);
    EXPECT_NEAR(-6.0,C.get(3,2,6,2), 1E-10);
}

TEST(SparseTensor_TEST, COO_DataIO)
{
    COOTensor<double, 3> tensor1(10, 2, 3, 4);
    tensor1.add_element(3.2, 0, 0, 0);
    tensor1.add_element(1.3, 0, 1, 1);
    tensor1.add_element(-9.7,0, 1, 2);
    tensor1.add_element(-8.4,1, 2, 3);
    tensor1.write_to_file("synspt_2_3_4.tns");

    COOTensor<double, 3> tensor2(10, 2, 3, 4);   
    tensor2.read_from_file("synspt_2_3_4.tns");     
    
    EXPECT_EQ(tensor2.nnz(), 4);   // Number of non-zeros
    EXPECT_NEAR(3.2, tensor2.get(0,0,0), 1E-10);
    EXPECT_NEAR(1.3, tensor2.get(0,1,1), 1E-10);
    EXPECT_NEAR(-9.7,tensor2.get(0,1,2), 1E-10);
    EXPECT_NEAR(-8.4,tensor2.get(1,2,3), 1E-10);

    auto full_tensor = tensor2.to_dense();
    EXPECT_NEAR(3.2, full_tensor(0,0,0), 1E-10);
    EXPECT_NEAR(1.3, full_tensor(0,1,1), 1E-10);
    EXPECT_NEAR(-9.7,full_tensor(0,1,2), 1E-10);
    EXPECT_NEAR(-8.4,full_tensor(1,2,3), 1E-10);

    if (std::remove("synspt_2_3_4.tns") != 0) {
        throw std::runtime_error("Error deleting file: synspt_2_3_4.tns");
    }
}

TEST(SparseTensor_TEST, COO_RandomDataGen)
{
    COOTensor<double, 3> tensor1(120, 4, 5, 6);  // Initial capacity = 4x5x6
    COOTensor<double, 3> tensor2(90, 6, 5, 3);  // Initial capacity = 4x5x6

    // Uniform distribution (default)
    tensor1.generate_random(0.3);  // 0.3 density, uniform [-1, 1]

    // Uniform with custom range
    tensor2.generate_random(0.3, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0));

    auto tensor3 = tensor1.contract(tensor2, 2, 0);
    //tensor3.print();

    // Normal distribution
    //tensor.generate_random(0.3, Distribution::NORMAL, DistributionParams::normal(5.0, 2.0));  // mean=5.0, std dev=2.0

    // Standard normal distribution (mean=0, std_dev=1)
    //tensor.generate_random(0.3, Distribution::STANDARD_NORMAL);

    // Gamma distribution
    //tensor.generate_random(0.3, Distribution::GAMMA, DistributionParams::gamma(2.0, 1.0));  // shape=2.0, scale=1.0

    // 6. With specific seed for reproducibility
    //tensor.generate_random(0.3, Distribution::NORMAL, DistributionParams::normal(0.0, 1.0), 42);  // seed=42
}

TEST(SparseTensor_TEST, COO_TTContraction1)
{
    // Create TT-cores with dimensions:
    // G1(1,n1,r1), G2(r1,n2,r2), G3(r2,n3,1)
    COOTensor<float, 2> G1(100, 4, 3);    // rank0=1, n1=4, rank1=3
    COOTensor<float, 3> G2(100, 3, 5, 2); // rank1=3, n2=5, rank2=2
    COOTensor<float, 2> G3(100, 2, 6);    // rank2=2, n3=6, rank3=1
    
    // Fill cores with some values
    G1.add_element(1.0f, 0, 0);
    G1.add_element(2.0f, 1, 1);
    
    G2.add_element(3.0f, 0, 0, 0);
    G2.add_element(4.0f, 1, 1, 0);
    
    G3.add_element(5.0f, 0, 0);
    G3.add_element(6.0f, 0, 1);
    
    // Contract the entire train
    // Result will be a tensor of shape (4,5,6) 
    auto result = SparseTTtoTensor<float>(G1, G2, G3);

    // Check the correctness
    EXPECT_EQ(result.nnz(), 4);   // Number of non-zeros
    EXPECT_NEAR(15.0f, result.get(0,0,0), 1E-8);
    EXPECT_NEAR(18.0f, result.get(0,0,1), 1E-8);
    EXPECT_NEAR(40.0f, result.get(1,1,0), 1E-8);
    EXPECT_NEAR(48.0f, result.get(1,1,1), 1E-8);
}

TEST(SparseTensor_TEST, COO_TTContraction2)
{
    // Create TT-cores with dimensions:
    COOTensor<double, 2> G1(100, 2, 5);    
    COOTensor<double, 3> G2(100, 5, 4, 3); 
    COOTensor<double, 3> G3(100, 3, 6, 2);
    COOTensor<double, 2> G4(100, 2, 10); 
    
    // Fill cores with some values
    G1.add_element(-1.0,1, 4);
    G1.add_element(2.0, 1, 1);
    G1.add_element(4.2, 1, 2);

    G2.add_element(3.0, 0, 3, 0);
    G2.add_element(4.0, 4, 1, 2);
    
    G3.add_element(5.4, 0, 0, 0);
    G3.add_element(-2.0,0, 4, 1);
    G3.add_element(3.0, 1, 1, 0);
    G3.add_element(1.1, 2, 3, 1);

    G4.add_element(-4.7,0, 4);
    G4.add_element(8.6, 1, 8);
    G4.add_element(10.0,1, 7);

    // Contract the entire train
    auto result = SparseTTtoTensor<double>(G1, G2, G3, G4);       

    // Check the correctness
    EXPECT_EQ(result.nnz(), 2);   // Number of non-zeros
    EXPECT_NEAR(-37.84,result.get(1,1,3,8), 1E-10);
    EXPECT_NEAR(-44.0, result.get(1,1,3,7), 1E-10);
}

TEST(SparseTensor_TEST, COO_TensorAddition1)
{
    // Create two 3x3x3 tensors with initial capacity of 5 elements each
    std::array<size_t, 3> dimensions = {3, 3, 3};
    COOTensor<double, 3> tensor1(5, dimensions);
    COOTensor<double, 3> tensor2(5, dimensions);

    // Add some elements to tensor1
    std::array<size_t, 3> idx1 = {0, 0, 0};
    std::array<size_t, 3> idx2 = {1, 1, 1};
    std::array<size_t, 3> idx3 = {2, 2, 2};
    tensor1.add_element_array(1.5, idx1);
    tensor1.add_element_array(2.0, idx2);
    tensor1.add_element_array(3.5, idx3);

    // Add some elements to tensor2
    std::array<size_t, 3> idx4 = {0, 0, 0};  // This will add to tensor1's (0,0,0)
    std::array<size_t, 3> idx5 = {1, 1, 1};  // This will add to tensor1's (1,1,1)
    std::array<size_t, 3> idx6 = {2, 1, 0};  // This is a new position
    tensor2.add_element_array(0.5, idx4);
    tensor2.add_element_array(1.0, idx5);
    tensor2.add_element_array(2.5, idx6);

    //std::cout << "Tensor 1:\n";
    //tensor1.print();
    
    //std::cout << "\nTensor 2:\n";
    //tensor2.print();

    // Add the tensors
    auto result = tensor1 + tensor2;
    //std::cout << "\nResult of addition:\n";
    //result.print();

    // Test the += operator
    tensor1 += tensor2;
    //std::cout << "\nTensor 1 after +=:\n";
    //tensor1.print();

    EXPECT_EQ(4, result.nnz());
    EXPECT_NEAR(2.0, result.get(0,0,0), 1E-10);
    EXPECT_NEAR(3.0, result.get(1,1,1), 1E-10);
    EXPECT_NEAR(2.5, result.get(2,1,0), 1E-10);
    EXPECT_NEAR(3.5, result.get(2,2,2), 1E-10);

    EXPECT_EQ(4, tensor1.nnz());
    EXPECT_NEAR(2.0, tensor1.get(0,0,0), 1E-10);
    EXPECT_NEAR(3.0, tensor1.get(1,1,1), 1E-10);
    EXPECT_NEAR(2.5, tensor1.get(2,1,0), 1E-10);
    EXPECT_NEAR(3.5, tensor1.get(2,2,2), 1E-10);
}


TEST(SparseTensor_TEST, COO_TensorAddition2)
{
    std::array<size_t, 3> dimensions = {5, 5, 5};
    COOTensor<float, 3> tensor1(125, dimensions);
    COOTensor<float, 3> tensor2(125, dimensions);

    // Add some random elements to tensor1 and tensor2
    tensor1.generate_random(0.3, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 100);
    tensor2.generate_random(0.3, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 200);
   
    //std::cout << "Tensor 1:\n";
    //tensor1.print();
    //std::cout << "\nTensor 2:\n";
    //tensor2.print();

    // Add the tensors
    auto result = tensor1 + tensor2;
    //std::cout << "\nResult of addition:\n";
    //result.print();

    // Test the += operator
    tensor1 += tensor2;
    //std::cout << "\nTensor 1 after +=:\n";
    //tensor1.print();

    EXPECT_EQ(result.nnz(), tensor1.nnz());
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k)
                EXPECT_NEAR(result.get(i,j,k), tensor1.get(i,j,k), 1E-7);
}

TEST(SparseTensor_TEST, COO_RandomSparseSynTensor1)
{
    COOTensor<double, 2> G1(10, 3, 2);    
    COOTensor<double, 3> G2(24, 2, 4, 3);
    COOTensor<double, 2> G3(15, 3, 5);

    G1.generate_random(0.3, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 100);
    G2.generate_random(0.3, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 200);
    G3.generate_random(0.3, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 300);

    auto T = SparseTTtoTensor<double>(G1, G2, G3);
    //std::cout << "T:\n";
    //T.print();

    COOTensor<double, 3> noise(60, 3, 4, 5);
    noise.generate_random(0.5, Distribution::UNIFORM, DistributionParams::uniform(-0.001, 0.001), 150);

    T += noise;
    //std::cout << "T after noise:\n";
    //T.print();
}

TEST(SparseMat_TEST, hashMap_CPU) 
{
    SparseVector<double> vec(10, 2);  // size=10, initial capacity=2
    vec.set(1, 1.5);
    vec.set(3, 2.5);
    vec.set(5, 3.5);  // This will trigger resize
    //std::cout << "Printing vector:" << std::endl;
    //vec.print();

    COOMatrix_l1<double> mat(5,3);
    mat.set(0, 0, 1.0);
    mat.set(1, 2, 6.0);
    mat.set(4, 1, 3.0);
    mat.set(3, 2, 5.0);
    
    EXPECT_NEAR(mat.get(1, 1), 0.0, 1E-14);
    EXPECT_NEAR(mat.get(0, 1), 0.0, 1E-14);
    EXPECT_NEAR(mat.get(4, 1), 3.0, 1E-14);
    EXPECT_NEAR(mat.get(3, 2), 5.0, 1E-14);
}