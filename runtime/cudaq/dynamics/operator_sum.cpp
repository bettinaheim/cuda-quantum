/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

#include <iostream>
#include <set>

namespace cudaq {

/// Operator sum constructor given a vector of product operators.
operator_sum::operator_sum(const std::vector<product_operator> &terms)
    : m_terms(terms) {}

std::tuple<std::vector<scalar_operator>, std::vector<elementary_operator>>
operator_sum::m_canonicalize_product(product_operator &prod) const {
  std::vector<int> all_degrees;
  for (auto op : prod.m_elementary_ops) {
    for (auto degree : op.degrees)
      all_degrees.push_back(degree);
  }
  auto scalars = prod.m_scalar_ops;
  auto non_scalars = prod.m_elementary_ops;

  std::set<int> unique_degrees(all_degrees.begin(), all_degrees.end());

  if (all_degrees.size() == unique_degrees.size()) {
    // Each operator acts on different degrees of freedom; they
    // hence commute and can be reordered arbitrarily.
    /// FIXME: Doing nothing for now
    // std::sort(non_scalars.begin(), non_scalars.end(), [](auto op){ return
    // op.degrees; })
  } else {
    // Some degrees exist multiple times; order the scalars, identities,
    // and zeros, but do not otherwise try to reorder terms.
    std::vector<elementary_operator> zero_ops;
    std::vector<elementary_operator> identity_ops;
    std::vector<elementary_operator> non_commuting;
    for (auto op : non_scalars) {
      if (op.id == "zero")
        zero_ops.push_back(op);
      if (op.id == "identity")
        identity_ops.push_back(op);
      if (op.id != "zero" || op.id != "identity")
        non_commuting.push_back(op);
    }

    /// FIXME: Not doing the same sorting we do in python yet
    std::vector<elementary_operator> sorted_non_scalars;
    sorted_non_scalars.insert(sorted_non_scalars.end(), zero_ops.begin(),
                              zero_ops.end());
    sorted_non_scalars.insert(sorted_non_scalars.end(), identity_ops.begin(),
                              identity_ops.end());
    sorted_non_scalars.insert(sorted_non_scalars.end(), non_commuting.begin(),
                              non_commuting.end());
    non_scalars = sorted_non_scalars;
  }
  auto merged_scalar = scalar_operator(1.0);
  for (auto scalar : scalars) {
    merged_scalar *= scalar;
  }
  scalars = {merged_scalar};
  return std::make_tuple(scalars, non_scalars);
}

std::tuple<std::vector<scalar_operator>, std::vector<elementary_operator>>
operator_sum::m_canonical_terms() const {
  /// FIXME: Not doing the same sorting we do in python yet
  std::tuple<std::vector<scalar_operator>, std::vector<elementary_operator>> result;
  std::vector<scalar_operator> scalars;
  std::vector<elementary_operator> elementary_ops;
  for (auto term : m_terms) {
    auto canon_term = m_canonicalize_product(term);
    auto canon_scalars = std::get<0>(canon_term);
    auto canon_elementary = std::get<1>(canon_term);
    scalars.insert(scalars.end(), canon_scalars.begin(), canon_scalars.end());
    canon_elementary.insert(canon_elementary.end(), canon_elementary.begin(), canon_elementary.end());
  }
  return std::make_tuple(scalars, elementary_ops);
}

// Creates a new instance where all sub-terms are sorted in canonical order.
// The new instance is equivalent to the original one, meaning it has the same
// effect on any quantum system for any set of parameters.
/// FIXME: Verify this function via testing.
operator_sum operator_sum::canonicalize() const {
  std::vector<product_operator> canonical_terms;
  auto _canonical_terms = m_canonical_terms();
  auto _canonical_scalars = std::get<0>(_canonical_terms);
  auto _canonical_non_scalars = std::get<1>(_canonical_terms);
  for (auto operators : _canonical_scalars) {
    canonical_terms.push_back(product_operator({operators}, {}));
  }
  for (auto operators : _canonical_non_scalars) {
    canonical_terms.push_back(product_operator({}, {operators}));
  }
  return operator_sum(canonical_terms);
}

// bool operator_sum::operator==(const operator_sum &other) const {
//   return m_canonical_terms() == other.m_canonical_terms();
// }

// Arithmetic operators
operator_sum operator_sum::operator+(const operator_sum &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.insert(combined_terms.end(),
                        std::make_move_iterator(other.m_terms.begin()),
                        std::make_move_iterator(other.m_terms.end()));
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator-(const operator_sum &other) const {
  return *this + (-1 * other);
}

operator_sum operator_sum::operator-=(const operator_sum &other) {
  *this = *this - other;
  return *this;
}

operator_sum operator_sum::operator+=(const operator_sum &other) {
  *this = *this + other;
  return *this;
}

operator_sum operator_sum::operator*(operator_sum &other) const {
  auto self_terms = m_terms;
  std::vector<product_operator> product_terms;
  auto other_terms = other.m_terms;
  for (auto &term : self_terms) {
    for (auto &other_term : other_terms) {
      product_terms.push_back(term * other_term);
    }
  }
  return operator_sum(product_terms);
}

operator_sum operator_sum::operator*=(operator_sum &other) {
  *this = *this * other;
  return *this;
}

operator_sum operator_sum::operator*(const scalar_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator+(const scalar_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(product_operator({other}, {}));
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator-(const scalar_operator &other) const {
  return *this + (-1.0 * other);
}

operator_sum operator_sum::operator*=(const scalar_operator &other) {
  *this = *this * other;
  return *this;
}

operator_sum operator_sum::operator+=(const scalar_operator &other) {
  *this = *this + other;
  return *this;
}

operator_sum operator_sum::operator-=(const scalar_operator &other) {
  *this = *this - other;
  return *this;
}

operator_sum operator_sum::operator*(std::complex<double> other) const {
  return *this * scalar_operator(other);
}

operator_sum operator_sum::operator+(std::complex<double> other) const {
  return *this + scalar_operator(other);
}

operator_sum operator_sum::operator-(std::complex<double> other) const {
  return *this - scalar_operator(other);
}

operator_sum operator_sum::operator*=(std::complex<double> other) {
  *this *= scalar_operator(other);
  return *this;
}

operator_sum operator_sum::operator+=(std::complex<double> other) {
  *this += scalar_operator(other);
  return *this;
}

operator_sum operator_sum::operator-=(std::complex<double> other) {
  *this -= scalar_operator(other);
  return *this;
}

operator_sum operator_sum::operator*(double other) const {
  return *this * scalar_operator(other);
}

operator_sum operator_sum::operator+(double other) const {
  return *this + scalar_operator(other);
}

operator_sum operator_sum::operator-(double other) const {
  return *this - scalar_operator(other);
}

operator_sum operator_sum::operator*=(double other) {
  *this *= scalar_operator(other);
  return *this;
}

operator_sum operator_sum::operator+=(double other) {
  *this += scalar_operator(other);
  return *this;
}

operator_sum operator_sum::operator-=(double other) {
  *this -= scalar_operator(other);
  return *this;
}

operator_sum operator*(std::complex<double> other, operator_sum self) {
  return scalar_operator(other) * self;
}

operator_sum operator+(std::complex<double> other, operator_sum self) {
  return scalar_operator(other) + self;
}

operator_sum operator-(std::complex<double> other, operator_sum self) {
  return scalar_operator(other) - self;
}

operator_sum operator*(double other, operator_sum self) {
  return scalar_operator(other) * self;
}

operator_sum operator+(double other, operator_sum self) {
  return scalar_operator(other) + self;
}

operator_sum operator-(double other, operator_sum self) {
  return scalar_operator(other) - self;
}

operator_sum operator_sum::operator+(const product_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(other);
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator+=(const product_operator &other) {
  *this = *this + other;
  return *this;
}

operator_sum operator_sum::operator-(const product_operator &other) const {
  return *this + (-1. * other);
}

operator_sum operator_sum::operator-=(const product_operator &other) {
  *this = *this - other;
  return *this;
}

operator_sum operator_sum::operator*(const product_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator*=(const product_operator &other) {
  *this = *this * other;
  return *this;
}

operator_sum operator_sum::operator+(const elementary_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(product_operator({}, {other}));
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator-(const elementary_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back((-1. * other));
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator*(const elementary_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator+=(const elementary_operator &other) {
  *this = *this + product_operator({}, {other});
  return *this;
}

operator_sum operator_sum::operator-=(const elementary_operator &other) {
 *this = *this - product_operator({}, {other});
  return *this;
}

operator_sum operator_sum::operator*=(const elementary_operator &other) {
  *this = *this * other;
  return *this;
}

cudaq::matrix_2 operator_sum::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) {
  std::set<int> degrees_set;
  for (auto op : m_terms) {
    for (auto degree : op.degrees()) {
      std::cout << "degree = " << degree << "\n";
      degrees_set.insert(degree);
    }
  }
  std::vector<int> degrees(degrees_set.begin(), degrees_set.end());

  // We need to make sure all matrices are of the same size to sum them up.
  auto paddedTerm = [&](product_operator term) {
    std::vector<int> op_degrees;
    for (auto op : term.m_elementary_ops) {
      for (auto degree : op.degrees)
        op_degrees.push_back(degree);
    }
    for (auto degree : degrees) {
      auto it = std::find(op_degrees.begin(), op_degrees.end(), degree);
      if (it == op_degrees.end())
        term *= elementary_operator::identity(degree);
    }
    return term;
  };

  auto sum = EvaluatedMatrix();
  if (pad_terms) {
    sum = EvaluatedMatrix(degrees, paddedTerm(m_terms[0])
                                       .m_evaluate(arithmetics, dimensions,
                                                   parameters, pad_terms));
    for (auto term_idx = 1; term_idx < m_terms.size(); ++term_idx) {
      auto term = m_terms[term_idx];
      auto eval = paddedTerm(term).m_evaluate(arithmetics, dimensions,
                                              parameters, pad_terms);
      sum = arithmetics.add(sum, EvaluatedMatrix(degrees, eval));
    }
  } else {
    sum =
        EvaluatedMatrix(degrees, m_terms[0].m_evaluate(arithmetics, dimensions,
                                                       parameters, pad_terms));
    for (auto term_idx = 1; term_idx < m_terms.size(); ++term_idx) {
      auto term = m_terms[term_idx];
      auto eval =
          term.m_evaluate(arithmetics, dimensions, parameters, pad_terms);
      sum = arithmetics.add(sum, EvaluatedMatrix(degrees, eval));
    }
  }
  return sum.matrix();
}

matrix_2 operator_sum::to_matrix(
    const std::map<int, int> &dimensions,
    const std::map<std::string, std::complex<double>> &parameters) {
  /// FIXME: Not doing any conversion to spin op yet.
  return m_evaluate(MatrixArithmetics(dimensions, parameters), dimensions,
                    parameters);
  ;
}

// std::string operator_sum::to_string() const {
//   std::string result;
//   // for (const auto &term : m_terms) {
//   //   result += term.to_string() + " + ";
//   // }
//   // // Remove last " + "
//   // if (!result.empty())
//   //   result.pop_back();
//   return result;
// }

} // namespace cudaq