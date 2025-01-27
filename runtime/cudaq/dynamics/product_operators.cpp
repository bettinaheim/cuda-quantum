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

product_operator::product_operator(
    std::vector<scalar_operator> scalars,
    std::vector<elementary_operator> atomic_operators)
    : m_scalar_ops(scalars), m_elementary_ops(atomic_operators) {}

product_operator::product_operator(product_operator &other) : m_scalar_ops(other.m_scalar_ops), m_elementary_ops(other.m_elementary_ops) {}
product_operator::product_operator(const product_operator &other) : m_scalar_ops(other.m_scalar_ops), m_elementary_ops(other.m_elementary_ops) {}

// Degrees property
std::vector<int> product_operator::degrees() const {
  std::set<int> unique_degrees;
  for (const auto &term : m_elementary_ops)
    unique_degrees.insert(term.degrees.begin(), term.degrees.end());

  // Erase any `-1` degree values that may have come from scalar operators.
  auto it = unique_degrees.find(-1);
  if (it != unique_degrees.end()) {
    unique_degrees.erase(it);
  }
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

operator_sum product_operator::operator+(scalar_operator &other) {
  return operator_sum({*this, product_operator({other}, {})});
}

operator_sum product_operator::operator-(scalar_operator &other) {
  return operator_sum({*this, -1. * product_operator({other}, {})});
}

product_operator product_operator::operator*(scalar_operator &other) {
  std::vector<scalar_operator> combined_scalars = m_scalar_ops;
  combined_scalars.push_back(other);  
  return product_operator(combined_scalars, m_elementary_ops);
}

void product_operator::operator*=(scalar_operator other) {
  m_scalar_ops.push_back(other);
}

operator_sum product_operator::operator+(std::complex<double> other) {
  return operator_sum({*this, product_operator({scalar_operator(other)}, {})});
}

operator_sum product_operator::operator-(std::complex<double> other) {
  return operator_sum({*this, product_operator({scalar_operator(-1.0), scalar_operator(other)}, {})});
}

product_operator product_operator::operator*(std::complex<double> other) {
  std::vector<scalar_operator> combined_scalars = m_scalar_ops;
  std::cout << "\n scalars size = " << m_scalar_ops.size() << "\n";
  std::cout << "\n elementary size = " << m_elementary_ops.size() << "\n";
  auto op = scalar_operator(other);
  combined_scalars.push_back(op);  
  std::cout << "\n combined scalars size = " << combined_scalars.size() << "\n";
  auto result = product_operator(combined_scalars, m_elementary_ops);
  std::cout << "\n result elementary size = " << result.m_elementary_ops.size() << "\n";
  std::cout << "\n result scalar size = " << result.m_scalar_ops.size() << "\n";
  return result;
}

void product_operator::operator*=(std::complex<double> other) {
  m_scalar_ops.push_back(scalar_operator(other));
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
  return operator_sum({*this, product_operator({scalar_operator(other)}, {})});
}

operator_sum product_operator::operator-(double other) {
  return operator_sum({*this, product_operator({scalar_operator(-1.0), scalar_operator(other)}, {})});
}

product_operator product_operator::operator*(double other) {
  std::vector<scalar_operator> combined_scalars = m_scalar_ops;
  combined_scalars.push_back(scalar_operator(other));  
  return product_operator(combined_scalars, m_elementary_ops);
}

void product_operator::operator*=(double other) {
  m_scalar_ops.push_back(scalar_operator(other));
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

operator_sum product_operator::operator+(product_operator &other) {
  std::cout << "\n line 142, product + product \n";
  return operator_sum({*this, other});
}

/// FIXME: Need to confirm I've fixed a previous bug here.
operator_sum product_operator::operator-(product_operator &other) {
  std::cout << "\n line 148, product + product \n";
  return operator_sum({*this, (-1. * other)});
}

product_operator product_operator::operator*(product_operator &other) {
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

product_operator product_operator::operator*=(product_operator &other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(elementary_operator &other) {
  std::vector<elementary_operator> _other = {other};
  return operator_sum({*this, product_operator({}, _other)});
}

operator_sum product_operator::operator-(elementary_operator &other) {
  std::vector<elementary_operator> _other = {other};
  return operator_sum({*this, product_operator({scalar_operator(-1.0)}, _other)});
}

product_operator product_operator::operator*(elementary_operator &other) {
  std::vector<elementary_operator> combined_elementary = m_elementary_ops;
  combined_elementary.push_back(other);  
  return product_operator(m_scalar_ops, combined_elementary);
}

void product_operator::operator*=(elementary_operator other) {
  m_elementary_ops.push_back(other);
}

operator_sum product_operator::operator+(operator_sum &other) {
  std::vector<product_operator> other_terms = other.m_terms;
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

operator_sum product_operator::operator-(operator_sum &other) {
  auto negative_other = (-1. * other);
  std::vector<product_operator> other_terms = negative_other.m_terms;
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

operator_sum product_operator::operator*(operator_sum &other) {
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
  std::complex<double> merged_scalar = 1.0+0.0j;
  std::cout << "\n number of scalar ops to merge = " << m_scalar_ops.size() << "\n";
  for (auto &scalar : m_scalar_ops) {
    std::cout << "\n merging in " << scalar.m_name << "\n";
    merged_scalar *= scalar.evaluate(parameters);
  }
  return merged_scalar * result;
}

} // namespace cudaq