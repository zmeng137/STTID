// dtensor.h - Toolkit for dense tensor operations
#ifndef DENSE_T_H
#define DENSE_T_H

// NOTE: Tblis should be always included before core.h
#include <tblis/tblis.h>
#include "core.h"

namespace denseT {

// Frobenious norm
template<class T>
T FrobNorm(tblis::tensor<T> tensor)
{
    auto shape = tensor.lengths(); 
    int dim = shape.size();       
    int len = 1;
    double fnorm = 0.0;
    for (int i=0; i<dim; ++i) len *= shape[i];
    auto data = tensor.data();
    for (int i=0; i<len; ++i) fnorm += data[i] * data[i];
    fnorm = std::sqrt(fnorm);
    return fnorm;
}

// Size..
template<class T>
size_t GetSize(tblis::tensor<T> tensor)
{
    auto shape = tensor.lengths(); 
    int dim = shape.size();       
    size_t len = 1;
    for (int i=0; i<dim; ++i) len *= shape[i];
    return len;
}

// Arbitrary-order norm
template<class T>
double Norm(tblis::tensor<T> tensor, int mode)
{
    if (mode != 1 && mode != 2) 
        throw std::invalid_argument("Incorrect mode! Mode should be either 1 (max norm) or 2 (frob2 norm).");
    double norm = 0.0;
    size_t N = GetSize(tensor);
    if (mode == 1) {
        // Mode 1: max norm
        auto data = tensor.data();
        for (size_t i = 0; i < N; ++i) 
            norm = std::max(std::abs(data[i]), norm);
    } else if (mode == 2) {
        // Mode 2: Frobenious 2 norm 
        auto data = tensor.data();
        for (size_t i = 0; i < N; ++i) 
            norm += data[i] * data[i];
        norm = std::sqrt(norm);
    }    
    return norm;
}

inline std::string generateLetters(char offset, int n) {
    std::string result;
    for (int i = 0; i < n; ++i) {
        result += offset + i;  // Append the next letter in the sequence
    }
    return result;
}

// Tensor train contraction (TT_to_Tensor)
template<class T>
tblis::tensor<T> TT_Contraction_dense(std::vector<tblis::tensor<T>> ttList)
{
    int recon_dim = ttList.size();
    tblis::tensor<T> tensor = ttList[0];
    for (int i = 1; i < recon_dim; ++i) {
        auto factor = ttList[i];
        auto lshape = tensor.lengths();
        auto rshape = factor.lengths();
        int ldim = lshape.size();
        int rdim = rshape.size();        
        MArray::len_vector ishape(ldim + rdim - 2);

        for (int i = 0; i < ldim + rdim - 2; ++i) {
            if (i < ldim - 1) {
                ishape[i] = lshape[i];
            } else {
                ishape[i] = rshape[i - ldim + 2];
            }
        }
        std::string aidx = generateLetters('a', ldim);
        std::string bidx = generateLetters(aidx.back(), rdim);
        std::string cidx = generateLetters('a', ldim + rdim - 1);
        cidx.erase(ldim-1, 1);

        tblis::tensor<T> inter_tensor(ishape);
        tblis::mult<T>(1.0, tensor, aidx.c_str(), factor, bidx.c_str(), 0.0, inter_tensor, cidx.c_str());  
        tensor.resize(ishape); 
        tensor = inter_tensor;
    }
    return tensor;
}

// Tblis-style synthetic dense tensor generator
template<class T>
tblis::tensor<T> SyntheticTenGen(std::initializer_list<int> tShape, std::initializer_list<int> tRank)
{
    if (tRank.size() != tShape.size() - 1) {
        throw std::invalid_argument("Invalid input tShape or tRank!");
    }
    
    std::vector<int> vec_tShape = tShape;
    std::vector<int> vec_tRank = tRank;
    vec_tRank.push_back(1);
    vec_tRank.insert(vec_tRank.begin(), 1);
    int dim = vec_tShape.size();
    std::vector<tblis::tensor<T>> ttList;
    
    // Tensor train generation
    for (int i = 0; i < dim; ++i) {
        int n = vec_tRank[i] * vec_tShape[i] * vec_tRank[i+1];
        tblis::tensor<T> factor({vec_tRank[i], vec_tShape[i], vec_tRank[i+1]});
        if (i == 0) { 
            factor.resize({vec_tShape[i], vec_tRank[i+1]}); }
        else if (i == dim - 1) { 
            factor.resize({vec_tRank[i], vec_tShape[i]}); }
        
        // Generate the random data for the tensor factor
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10,10);
        T* randSeq = new T[n];
        for (int i = 0; i < n; ++i) randSeq[i] = dis(gen);
        std::copy(randSeq, randSeq + n, factor.data());       
        ttList.push_back(factor);
        delete[] randSeq;    
    }
    // TT contraction -> Synthetic output tensor
    auto synTensor = denseT::TT_Contraction_dense(ttList);
    return synTensor;
}

// Tblis-style synthetic sparse tensor generator
template<class T>
tblis::tensor<T> SyntheticSparseTenGen(std::initializer_list<int> tShape, std::initializer_list<int> tRank,
                                       std::initializer_list<double> tDensity)
{
    if (tRank.size() != tShape.size() - 1) 
        throw std::invalid_argument("Invalid input tShape or tRank!");
    if (tShape.size() != tDensity.size())
        throw std::invalid_argument("Invalid input tShape or density!");
    
    std::vector<int> vec_tShape = tShape;
    std::vector<int> vec_tRank = tRank;
    std::vector<double> vec_density = tDensity;
    vec_tRank.push_back(1);
    vec_tRank.insert(vec_tRank.begin(), 1);
    int dim = vec_tShape.size();
    std::vector<tblis::tensor<T>> ttList;
    
    // Tensor train generation
    for (int i = 0; i < dim; ++i) {
        int n = vec_tRank[i] * vec_tShape[i] * vec_tRank[i+1];
        tblis::tensor<T> factor({vec_tRank[i], vec_tShape[i], vec_tRank[i+1]});
        if (i == 0) { 
            factor.resize({vec_tShape[i], vec_tRank[i+1]}); }
        else if (i == dim - 1) { 
            factor.resize({vec_tRank[i], vec_tShape[i]}); }
        
        // Generate the random data for the tensor factor
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-10,10);           
        std::uniform_real_distribution<double> dist(0.0, 1.0);   
        double density = vec_density[i];
        double spThres;
        T* randSeq = new T[n];
        for (int i = 0; i < n; ++i) {
            spThres = dist(gen);
            if (spThres < density) {
                randSeq[i] = dis(gen);
            } else {
                randSeq[i] = 0.0;
            }
        }        
        std::copy(randSeq, randSeq + n, factor.data());       
        ttList.push_back(factor);
        delete[] randSeq;    
    }
    // TT contraction -> Synthetic output tensor
    auto synTensor = denseT::TT_Contraction_dense(ttList);
    return synTensor;
}

// Evaluation of the recontruction error
template<class T>
double NormError(tblis::tensor<T> tensorA, tblis::tensor<T> tensorB, int mode, bool relative)
{
    size_t N = GetSize(tensorA);
    if (N != GetSize(tensorB))
        throw std::invalid_argument("Size of tensorA != Size of tensorB!");
    if (mode != 1 && mode != 2) 
        throw std::invalid_argument("Incorrect mode! Mode should be either 1 (max-norm error) or 2 (frob-norm error).");
    
    double error = 0.0;
    if (mode == 1) {
        // Mode 1: max-norm error
        auto tensorA_data = tensorA.data();
        auto tensorB_data = tensorB.data();
        for (size_t i = 0; i < N; ++i) 
            error = std::max(error, std::abs(tensorA_data[i] - tensorB_data[i]));      
    } else if (mode == 2) {
        // Mode 2: frobenious norm-2 error
        auto tensorA_data = tensorA.data();
        auto tensorB_data = tensorB.data();
        for (size_t i = 0; i < N; ++i) {
            auto diff = tensorA_data[i] - tensorB_data[i];
            error += diff * diff;
        }
        error = std::sqrt(error);
    }
    if (relative) 
        error /= Norm(tensorA, mode);
    
    return error;
}
}

#endif 