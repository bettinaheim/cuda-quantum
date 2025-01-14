/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "helpers.cpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>
#include <set>

namespace cudaq {

/// Product Operator constructors.
product_operator::product_operator(
    std::vector<std::variant<scalar_operator, elementary_operator>>
        atomic_operators)
    : m_terms(atomic_operators) {
  for (auto term : m_terms) {
    if (std::holds_alternative<scalar_operator>(term)) {
      auto cast_term = std::get<scalar_operator>(term);
      m_scalar_ops.push_back(cast_term);
    } else if (std::holds_alternative<elementary_operator>(term)) {
      auto cast_term = std::get<elementary_operator>(term);
      m_elementary_ops.push_back(cast_term);
    }
  }
}

product_operator::product_operator(
    std::vector<scalar_operator> scalars,
    std::vector<elementary_operator> atomic_operators)
    : m_scalar_ops(scalars), m_elementary_ops(atomic_operators) {}

// Degrees property
std::vector<int> product_operator::degrees() const {
  std::set<int> unique_degrees;
  // The variant type makes it difficult
  auto beginFunc = [](auto &&t) { return t.degrees.begin(); };
  auto endFunc = [](auto &&t) { return t.degrees.end(); };
  for (const auto &term : m_terms) {
    unique_degrees.insert(std::visit(beginFunc, term),
                          std::visit(endFunc, term));
  }
  // Erase any `-1` degree values that may have come from scalar operators.
  auto it = unique_degrees.find(-1);
  if (it != unique_degrees.end()) {
    unique_degrees.erase(it);
  }
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

operator_sum product_operator::operator+(scalar_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, product_operator(_other)});
}

operator_sum product_operator::operator-(scalar_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, -1. * product_operator(_other)});
}

product_operator product_operator::operator*(scalar_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.push_back(other);
  return product_operator(combined_terms);
}

product_operator product_operator::operator*=(scalar_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(std::complex<double> other) {
  return *this + scalar_operator(other);
}

operator_sum product_operator::operator-(std::complex<double> other) {
  return *this - scalar_operator(other);
}

product_operator product_operator::operator*(std::complex<double> other) {
  return *this * scalar_operator(other);
}

product_operator product_operator::operator*=(std::complex<double> other) {
  *this = *this * scalar_operator(other);
  return *this;
}

operator_sum operator+(std::complex<double> other, product_operator self) {
  return operator_sum({scalar_operator(other), self});
}

operator_sum operator-(std::complex<double> other, product_operator self) {
  return scalar_operator(other) - self;
}

product_operator operator*(std::complex<double> other, product_operator self) {
  return scalar_operator(other) * self;
}

operator_sum product_operator::operator+(double other) {
  return *this + scalar_operator(other);
}

operator_sum product_operator::operator-(double other) {
  return *this - scalar_operator(other);
}

product_operator product_operator::operator*(double other) {
  return *this * scalar_operator(other);
}

product_operator product_operator::operator*=(double other) {
  *this = *this * scalar_operator(other);
  return *this;
}

operator_sum operator+(double other, product_operator self) {
  return operator_sum({scalar_operator(other), self});
}

operator_sum operator-(double other, product_operator self) {
  return scalar_operator(other) - self;
}

product_operator operator*(double other, product_operator self) {
  return scalar_operator(other) * self;
}

operator_sum product_operator::operator+(product_operator other) {
  return operator_sum({*this, other});
}

operator_sum product_operator::operator-(product_operator other) {
  return operator_sum({*this, (-1. * other)});
}

product_operator product_operator::operator*(product_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.insert(combined_terms.end(),
                        std::make_move_iterator(other.m_terms.begin()),
                        std::make_move_iterator(other.m_terms.end()));
  return product_operator(combined_terms);
}

product_operator product_operator::operator*=(product_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(elementary_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, product_operator(_other)});
}

operator_sum product_operator::operator-(elementary_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, -1. * product_operator(_other)});
}

product_operator product_operator::operator*(elementary_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.push_back(other);
  return product_operator(combined_terms);
}

product_operator product_operator::operator*=(elementary_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(operator_sum other) {
  std::vector<product_operator> other_terms = other.m_terms;
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

operator_sum product_operator::operator-(operator_sum other) {
  auto negative_other = (-1. * other);
  std::vector<product_operator> other_terms = negative_other.m_terms;
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

operator_sum product_operator::operator*(operator_sum other) {
  std::vector<product_operator> other_terms = other.m_terms;
  for (auto &term : other_terms) {
    term = *this * term;
  }
  return operator_sum(other_terms);
}

cudaq::matrix_2 product_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {

  /// FIXME: implement me :-)
  return cudaq::matrix_2();
}

cudaq::matrix_2
_padded_op(cudaq::MatrixArithmetics arithmetics,
           std::variant<cudaq::scalar_operator, cudaq::elementary_operator> op,
           std::vector<int> degrees, std::map<int, int> dimensions,
           std::map<std::string, std::complex<double>> parameters) {
  auto beginFunc = [](auto &&t) { return t.degrees.begin(); };
  auto endFunc = [](auto &&t) { return t.degrees.end(); };
  auto getMatrix = [&](auto &&t) {
    return t.to_matrix(dimensions, parameters);
  };
  /// Creating the tensor product with op being last is most efficient.
  auto accumulate_ops = [&]() {
    std::vector<cudaq::matrix_2> result;
    for (const auto &degree : degrees) {
      if (std::find(std::visit(beginFunc, op), std::visit(endFunc, op),
                    degree) == std::visit(endFunc, op),
          degree) {
        result.push_back(
            arithmetics.evaluate(cudaq::elementary_operator::identity(degree))
                .matrix());
      }
      matrix_2 mat = std::visit(getMatrix, op);
      result.push_back(mat);
    }
    return result;
  };
  auto padded = accumulate_ops();
  /// FIXME: This directly uses cudaq::kronecker instead of the tensor method.
  /// I need to double check to make sure this gives the equivalent behavior
  /// to the method used in python.
  auto result = cudaq::kronecker(padded.begin(), padded.end());
  return result;
}

cudaq::matrix_2 product_operator::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) {
  if (pad_terms) {
    // Sorting the degrees to avoid unnecessary permutations during the padding.
    std::set<int> noncanon_set;
    for (auto op : m_elementary_ops) {
      for (auto degree : op.degrees) {
        noncanon_set.insert(degree);
      }
    }
    std::vector<int> noncanon_degrees(noncanon_set.begin(), noncanon_set.end());
    auto degrees = _OperatorHelpers::canonicalize_degrees(noncanon_degrees);
    auto evaluated =
        EvaluatedMatrix(degrees, _padded_op(arithmetics, m_elementary_ops[0],
                                            degrees, dimensions, parameters));

    for (auto op_idx = 1; op_idx <= m_elementary_ops.size(); ++op_idx) {
      auto op = m_elementary_ops[op_idx];
      if (op.degrees.size() != 1) {
        auto padded_mat =
            EvaluatedMatrix(degrees, _padded_op(arithmetics, op, degrees,
                                                dimensions, parameters));
        evaluated = arithmetics.mul(evaluated, padded_mat);
      }
    }
    /// FIXME: Need to multiply through by the scalar operators!!!!!
    return evaluated.matrix();
  } else {
    auto evaluated = arithmetics.evaluate(m_elementary_ops[0]);
    for (auto op_idx = 1; op_idx <= m_elementary_ops.size(); ++op_idx) {
      auto op = m_elementary_ops[op_idx];
      auto mat = op.to_matrix(dimensions, parameters);
      evaluated = arithmetics.mul(evaluated, EvaluatedMatrix(op.degrees, mat));
    }
    /// FIXME: Need to multiply through by the scalar operators!!!!!
    return evaluated.matrix();
  }
}

} // namespace cudaq