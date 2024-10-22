/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "common/EigenDense.h"

#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>

namespace cudaq {

/// Product Operator constructors.
product_operator::product_operator(std::vector<std::variant<scalar_operator, elementary_operator>> atomic_operators) : m_terms(atomic_operators) {}

complex_matrix kroneckerHelper(std::vector<complex_matrix> &matrices) {
  // essentially we pass in the list of elementary operators to
  // this function -- with lowest degree being leftmost -- then it computes the
  // kronecker product of all of them.
  auto kronecker = [](complex_matrix self, complex_matrix other) { 
    return self.kronecker(other);
  };

  return std::accumulate(begin(matrices), end(matrices), complex_matrix::identity(1,1), kronecker);
}

/// IMPLEMENT:
complex_matrix product_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {

  /// TODO: This initial logic may not be needed.
  // std::vector<int> degrees, levels;
  // for(std::map<int,int>::iterator it = dimensions.begin(); it != dimensions.end(); ++it) {
  //   degrees.push_back(it->first);
  //   levels.push_back(it->second);
  // }
  // // Calculate the size of the full Hilbert space of the given product operator.
  // int fullSize = std::accumulate(begin(levels), end(levels), 1, std::multiplies<int>());
  std::cout << "here 49\n";
  auto getDegrees = [](auto &&t){ return t.degrees; };
  auto getMatrix = [&](auto &&t){ 
    auto outMatrix = t.to_matrix(dimensions, parameters);
    std::cout << "dumping the outMatrix : \n";
    outMatrix.dump();
    return outMatrix;
  };
  std::vector<complex_matrix> matricesFullVectorSpace;
  for (auto &term : m_terms) {
    auto op_degrees = std::visit(getDegrees, term);
    std::cout << "here 58\n";
    // Keeps track of if we've already inserted the operator matrix
    // into the full list of matrices.
    bool alreadyInserted = false;
    std::vector<complex_matrix> matrixWithIdentities;
    /// General procedure for inserting identities:
    // * check if the operator acts on this degree by looking through `op_degrees`
    // * if not, insert an identity matrix of the proper level size
    // * if so, insert the matrix itself
    for (auto [degree,level] : dimensions) {
      std::cout << "here 68\n";
      auto it = std::find(op_degrees.begin(), op_degrees.end(), degree);
      if (it != op_degrees.end() && !alreadyInserted) {
        std::cout << "here 71\n";
        auto matrix = std::visit(getMatrix, term);
        std::cout << "here 75\n";
        matrixWithIdentities.push_back(matrix);
        std::cout << "here 77\n";
      }
      else {
        std::cout << "here 80\n";
        matrixWithIdentities.push_back(complex_matrix::identity(level,level));
      }
    }
    std::cout << "here 84\n";
    matricesFullVectorSpace.push_back(kroneckerHelper(matrixWithIdentities));
  }
  // Now just need to accumulate with matrix multiplication all of the
  // matrices in `matricesFullVectorSpace` -- they should all be the same size already.
  std::cout << "here 89\n";

  // temporary
  auto out = complex_matrix::identity(1,1);
  std::cout << "here 93\n";
  return out;
}

// Degrees property
std::vector<int> product_operator::degrees() const {
  std::set<int> unique_degrees;
  // The variant type makes it difficult 
  auto beginFunc = [](auto &&t){ return t.degrees.begin(); };
  auto endFunc = [](auto &&t){ return t.degrees.end(); };
  for (const auto &term : m_terms) {
    unique_degrees.insert(std::visit(beginFunc, term), std::visit(endFunc, term));
  }
  // Erase any `-1` degree values that may have come from scalar operators.
  auto it = unique_degrees.find(-1);
  if (it != unique_degrees.end()) {
      unique_degrees.erase(it);
  }
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}


  // operator_sum product_operator::operator+(std::complex<double> other);
  // operator_sum product_operator::operator-(std::complex<double> other);
  // product_operator product_operator::operator*(std::complex<double> other);
  // product_operator product_operator::operator*=(std::complex<double> other);
  // operator_sum product_operator::operator+(operator_sum other);
  // operator_sum product_operator::operator-(operator_sum other);
  // product_operator product_operator::operator*(operator_sum other);
  // product_operator product_operator::operator*=(operator_sum other);
  // operator_sum product_operator::operator+(scalar_operator other);
  // operator_sum product_operator::operator-(scalar_operator other);
  // product_operator product_operator::operator*(scalar_operator other);
  // product_operator product_operator::operator*=(scalar_operator other);
  // operator_sum product_operator::operator+(product_operator other);
  // operator_sum product_operator::operator-(product_operator other);
  // product_operator product_operator::operator*(product_operator other);
  // product_operator product_operator::operator*=(product_operator other);
  // operator_sum product_operator::operator+(elementary_operator other);
  // operator_sum product_operator::operator-(elementary_operator other);
  // product_operator product_operator::operator*(elementary_operator other);
  // product_operator product_operator::operator*=(elementary_operator other);





} // namespace cudaq