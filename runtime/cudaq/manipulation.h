/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "operators.h"
#include "utils/tensor.h"

namespace cudaq {

template <typename TEval>
class OperatorArithmetics {
public:
  /// @brief Accesses the relevant data to evaluate an operator expression
  /// in the leaf nodes, that is in elementary and scalar operators.
  TEval evaluate(std::variant<elementary_operator, scalar_operator> op);

  /// @brief Adds two operators that act on the same degrees of freedom.
  TEval add(TEval val1, TEval val2);

  /// @brief Multiplies two operators that act on the same degrees of freedom.
  TEval mul(TEval val1, TEval val2);

  /// @brief Computes the tensor product of two operators that act on different
  /// degrees of freedom.
  TEval tensor(TEval val1, TEval val2);
};

class EvaluatedMatrix {
  friend class MatrixArithmetics;

private:
  std::vector<int> m_degrees;
  matrix_2 m_matrix;

public:
  EvaluatedMatrix(std::vector<int> degrees, matrix_2 matrix)
      : m_degrees(degrees), m_matrix(matrix) {}

  /// @brief The degrees of freedom that the matrix of the evaluated value
  /// applies to.
  std::vector<int> degrees() { return m_degrees; }

  /// @brief The matrix representation of an evaluated operator, according
  /// to the sequence of degrees of freedom associated with the evaluated
  /// value.
  matrix_2 matrix() { return m_matrix; }
};

/// Encapsulates the functions needed to compute the matrix representation
/// of an operator expression.
class MatrixArithmetics : public OperatorArithmetics<EvaluatedMatrix> {
private:
  std::map<int, int> m_dimensions;
  std::map<std::string, std::complex<double>> m_parameters;
  std::vector<int> _compute_permutation(std::vector<int> op_degrees,
                                        std::vector<int> canon_degrees);
  std::tuple<matrix_2, std::vector<int>>
  _canonicalize(matrix_2 &op_matrix, std::vector<int> op_degrees);

public:
  MatrixArithmetics(std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> parameters)
      : m_dimensions(dimensions), m_parameters(parameters) {}

  // Computes the tensor product of two evaluate operators that act on
  // different degres of freedom using the kronecker product.
  EvaluatedMatrix tensor(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Multiplies two evaluated operators that act on the same degrees
  // of freedom.
  EvaluatedMatrix mul(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Adds two evaluated operators that act on the same degrees
  // of freedom.
  EvaluatedMatrix add(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Computes the matrix of an ElementaryOperator or ScalarOperator using its
  // `to_matrix` method.
  EvaluatedMatrix
  evaluate(std::variant<elementary_operator, scalar_operator> op);
};

} // namespace cudaq