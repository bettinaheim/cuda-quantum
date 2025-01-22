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

// /// Product Operator constructors.
// product_operator::product_operator(
//     std::vector<std::variant<scalar_operator, elementary_operator>>
//         atomic_operators)
//     : m_terms(atomic_operators) {
//   for (auto term : m_terms) {
//     if (std::holds_alternative<scalar_operator>(term)) {
//       auto cast_term = std::get<scalar_operator>(term);
//       m_scalar_ops.push_back(cast_term);
//     } else if (std::holds_alternative<elementary_operator>(term)) {
//       auto cast_term = std::get<elementary_operator>(term);
//       m_elementary_ops.push_back(cast_term);
//     }
//   }
// }

product_operator::product_operator(
    std::vector<scalar_operator> scalars,
    std::vector<elementary_operator> atomic_operators)
    : m_scalar_ops(scalars), m_elementary_ops(atomic_operators) {}

// Degrees property
std::vector<int> product_operator::degrees() const {
  std::set<int> unique_degrees;
  // // The variant type makes it difficult
  // auto beginFunc = [](auto &&t) { return t.degrees.begin(); };
  // auto endFunc = [](auto &&t) { return t.degrees.end(); };
  // for (const auto &term : m_terms) {
  //   unique_degrees.insert(std::visit(beginFunc, term),
  //                         std::visit(endFunc, term));
  // }
  for (const auto &term : m_elementary_ops)
    unique_degrees.insert(term.degrees.begin(), term.degrees.end());

  // Erase any `-1` degree values that may have come from scalar operators.
  auto it = unique_degrees.find(-1);
  if (it != unique_degrees.end()) {
    unique_degrees.erase(it);
  }
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

operator_sum product_operator::operator+(scalar_operator other) {
  std::vector<scalar_operator> _other = {other};
  return operator_sum({*this, product_operator(_other, {})});
}

operator_sum product_operator::operator-(scalar_operator other) {
  std::vector<scalar_operator> _other = {other};
  return operator_sum({*this, -1. * product_operator(_other, {})});
}

product_operator product_operator::operator*(scalar_operator other) {
  std::vector<scalar_operator> combined_scalars = m_scalar_ops;
  combined_scalars.push_back(other);  
  return product_operator(combined_scalars, m_elementary_ops);
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
  std::cout << "\n line 142, product + product \n";
  return operator_sum({*this, other});
}

/// FIXME: Need to confirm I've fixed a previous bug here.
operator_sum product_operator::operator-(product_operator other) {
  std::cout << "\n line 148, product + product \n";
  return operator_sum({*this, (-1. * other)});
}

product_operator product_operator::operator*(product_operator other) {
  std::vector<elementary_operator> combined_elementary = m_elementary_ops;
  std::vector<scalar_operator> combined_scalars = m_scalar_ops;
  combined_elementary.insert(combined_elementary.end(),
                        std::make_move_iterator(other.m_elementary_ops.begin()),
                        std::make_move_iterator(other.m_elementary_ops.end()));

  combined_scalars.insert(combined_scalars.end(),
                        std::make_move_iterator(other.m_scalar_ops.begin()),
                        std::make_move_iterator(other.m_scalar_ops.end()));
  return product_operator(combined_scalars, combined_elementary);
}

product_operator product_operator::operator*=(product_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(elementary_operator other) {
  std::vector<elementary_operator> _other = {other};
  return operator_sum({*this, product_operator({}, _other)});
}

operator_sum product_operator::operator-(elementary_operator other) {
  std::vector<elementary_operator> _other = {other};
  return operator_sum({*this, product_operator({scalar_operator(-1.0)}, _other)});
}

product_operator product_operator::operator*(elementary_operator other) {
  std::vector<elementary_operator> combined_elementary = m_elementary_ops;
  combined_elementary.push_back(other);  
  return product_operator(m_scalar_ops, combined_elementary);
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
  return m_evaluate(MatrixArithmetics(dimensions, parameters), dimensions,
                    parameters);
}

cudaq::matrix_2
_padded_op(cudaq::MatrixArithmetics arithmetics, cudaq::elementary_operator op,
           std::vector<int> degrees, std::map<int, int> dimensions,
           std::map<std::string, std::complex<double>> parameters) {
  /// Creating the tensor product with op being last is most efficient.
  std::vector<cudaq::matrix_2> padded;
  for (const auto &degree : degrees) {
    if (std::find(op.degrees.begin(), op.degrees.end(), degree) ==
            op.degrees.end(),
        degree) {
      padded.push_back(
          arithmetics.evaluate(cudaq::elementary_operator::identity(degree))
              .matrix());
    }
    matrix_2 mat = op.to_matrix(dimensions, parameters);
    padded.push_back(mat);
  }
  /// FIXME: This directly uses cudaq::kronecker instead of the tensor method.
  /// I need to double check to make sure this gives the equivalent behavior
  /// to the method used in python.
  return cudaq::kronecker(padded.begin(), padded.end());
  ;
}

cudaq::matrix_2 product_operator::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) {
  std::set<int> noncanon_set;
  for (const auto &op : m_elementary_ops) {
    for (const auto &degree : op.degrees) {
      noncanon_set.insert(degree);
    }
  }
  std::vector<int> noncanon_degrees(noncanon_set.begin(), noncanon_set.end());

  // Calculate the total dimensions of the Hilbert space to create our
  // identity matrix.
  auto full_hilbert_size = 1;
  for (const auto [degree, dimension] : dimensions)
    full_hilbert_size *= dimension;
  cudaq::matrix_2 result(full_hilbert_size, full_hilbert_size);
  // If this product operator consists only of scalar operator terms,
  // we will avoid all of the below logic and just return the scalar value
  // stored in an identity matrix spanning the full Hilbert space of the
  // provided `dimensions`.
  if (m_elementary_ops.size() > 0) {
    if (pad_terms) {
      // Sorting the degrees to avoid unnecessary permutations during the
      // padding.
      std::set<int> noncanon_set;
      for (const auto &op : m_elementary_ops) {
        for (const auto &degree : op.degrees) {
          noncanon_set.insert(degree);
        }
      }
      auto degrees = _OperatorHelpers::canonicalize_degrees(noncanon_degrees);
      auto evaluated =
          EvaluatedMatrix(degrees, _padded_op(arithmetics, m_elementary_ops[0],
                                              degrees, dimensions, parameters));

      for (auto op_idx = 1; op_idx < m_elementary_ops.size(); ++op_idx) {
        auto op = m_elementary_ops[op_idx];
        if (op.degrees.size() != 1) {
          auto padded_op_to_print =
              _padded_op(arithmetics, op, degrees, dimensions, parameters);
          auto padded_mat =
              EvaluatedMatrix(degrees, _padded_op(arithmetics, op, degrees,
                                                  dimensions, parameters));
          evaluated = arithmetics.mul(evaluated, padded_mat);
        }
      }
      result = evaluated.matrix();
    } else {
      auto evaluated = arithmetics.evaluate(m_elementary_ops[0]);
      for (auto op_idx = 1; op_idx < m_elementary_ops.size(); ++op_idx) {
        auto op = m_elementary_ops[op_idx];
        auto mat = op.to_matrix(dimensions, parameters);
        evaluated =
            arithmetics.mul(evaluated, EvaluatedMatrix(op.degrees, mat));
      }
      result = evaluated.matrix();
    }
  } else {
    result = cudaq::matrix_2::identity(full_hilbert_size);
  }
  // We will merge all of the scalar values stored in `m_scalar_ops`
  // into a single scalar value.
  std::cout << "\n merging the scalars in `product_operator::m_evaluate` \n";
  auto merged_scalar = scalar_operator(1.0);
  for (auto scalar : m_scalar_ops)
    merged_scalar *= scalar;
  return merged_scalar.evaluate(parameters) * result;
}

} // namespace cudaq