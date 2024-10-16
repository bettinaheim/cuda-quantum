/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "common/EigenDense.h"

#include <iostream>
#include <set>

namespace cudaq {

/// Product Operator constructors.
product_operator::product_operator(std::vector<std::variant<scalar_operator, elementary_operator>> atomic_operators) : m_terms(atomic_operators) {}

/// FIXME:
complex_matrix product_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {
  return m_generator(dimensions, parameters);
}

  // operator_sum operator+(std::complex<double> other);
  // operator_sum operator-(std::complex<double> other);
  // product_operator operator*(std::complex<double> other);
  // product_operator operator*=(std::complex<double> other);
  // operator_sum operator+(operator_sum other);
  // operator_sum operator-(operator_sum other);
  // product_operator operator*(operator_sum other);
  // product_operator operator*=(operator_sum other);
  // operator_sum operator+(scalar_operator other);
  // operator_sum operator-(scalar_operator other);
  // product_operator operator*(scalar_operator other);
  // product_operator operator*=(scalar_operator other);
  // operator_sum operator+(product_operator other);
  // operator_sum operator-(product_operator other);
  // product_operator operator*(product_operator other);
  // product_operator operator*=(product_operator other);
  // operator_sum operator+(elementary_operator other);
  // operator_sum operator-(elementary_operator other);
  // product_operator operator*(elementary_operator other);
  // product_operator operator*=(elementary_operator other);





} // namespace cudaq