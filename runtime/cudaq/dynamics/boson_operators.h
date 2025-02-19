/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <unordered_map>
#include <vector>

#include "cudaq/utils/tensor.h"
#include "cudaq/operators.h"

namespace cudaq {

template <typename HandlerTy> 
class product_operator;

template <typename HandlerTy> 
class operator_sum;

// FIXME: rename?
class boson_operator : public operator_handler{
template <typename T> friend class product_operator;

private:

  // Each boson operator is represented as number operators along with an
  // offset to add to each number operator, as well as an integer indicating
  // how many creation or annihilation terms follow the number operators.
  // See the implementation of the in-place multiplication to understand
  // the meaning and purpose of this representation. In short, this
  // representation allows us to perform a perfect in-place multiplication.
  int additional_terms;
  std::vector<int> number_offsets;
  int target;

  // 0 = I, ad = 1, a = 2, ada = 3
  boson_operator(int target, int op_code);

  std::string op_code_to_string() const;

  void inplace_mult(const boson_operator &other);

public:

  // read-only properties

  virtual std::string unique_id() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  virtual std::vector<int> degrees() const;

  // constructors and destructors

  boson_operator(int target);

  ~boson_operator() = default;

  // evaluations

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2 to_matrix(std::unordered_map<int, int> &dimensions,
                             const std::unordered_map<std::string, std::complex<double>> &parameters = {}) const;

  virtual std::string to_string(bool include_degrees) const;

  // comparisons

  bool operator==(const boson_operator &other) const;

  // defined operators

  static operator_sum<boson_operator> empty();
  static product_operator<boson_operator> identity();

  static product_operator<boson_operator> identity(int degree);
  static product_operator<boson_operator> create(int degree);
  static product_operator<boson_operator> annihilate(int degree);
  static product_operator<boson_operator> number(int degree);

  static operator_sum<boson_operator> position(int degree);
  static operator_sum<boson_operator> momentum(int degree);
};

} // namespace cudaq