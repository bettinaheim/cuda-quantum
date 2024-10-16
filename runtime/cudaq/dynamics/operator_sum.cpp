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

/// Operator sum constructor given a vector of product operators.
operator_sum::operator_sum(const std::vector<product_operator> &terms)
    : m_terms(terms) {}

// std::vector<std::tuple<scalar_operator, elementary_operator>>
// operator_sum::canonicalize_product(product_operator &prod) const {
//   std::vector<std::tuple<scalar_operator, elementary_operator>>
//       canonicalized_terms;

  // std::vector<int> all_degrees;
  // std::vector<scalar_operator> scalars;
  // std::vector<elementary_operator> non_scalars;

  // for (const auto &op : prod.get_terms()) {
  //   if (std::holds_alternative<scalar_operator>(op)) {
  //     scalars.push_back(*std::get<scalar_operator>(op));
  //   } else {
  //     non_scalars.push_back(*std::get<elementary_operator>(op));
  //     all_degrees.insert(all_degrees.end(),
  //                        std::get<elementary_operator>(op).degrees.begin(),
  //                        std::get<elementary_operator>(op).degrees.end());
  //   }
  // }

  // if (all_degrees.size() ==
  //     std::set<int>(all_degrees.begin(), all_degrees.end()).size()) {
  //   std::sort(non_scalars.begin(), non_scalars.end(),
  //             [](const elementary_operator &a, const elementary_operator &b) {
  //               return a.degrees < b.degrees;
  //             });
  // }

  // for (size_t i = 0; std::min(scalars.size(), non_scalars.size()); i++) {
  //   canonicalized_terms.push_back(std::make_tuple(scalars[i], non_scalars[i]));
  // }

//   return canonicalized_terms;
// }

// std::vector<std::tuple<scalar_operator, elementary_operator>>
// operator_sum::_canonical_terms() const {
//   std::vector<std::tuple<scalar_operator, elementary_operator>> terms;
//   // for (const auto &term : m_terms) {
//   //   auto canonicalized = canonicalize_product(term);
//   //   terms.insert(terms.end(), canonicalized.begin(), canonicalized.end());
//   // }

//   // std::sort(terms.begin(), terms.end(), [](const auto &a, const auto &b) {
//   //   // return std::to_string(product_operator(a)) <
//   //   //        std::to_string(product_operator(b));
//   //   return product_operator(a).to_string() < product_operator(b).to_string();
//   // });

//   return terms;
// }

// operator_sum operator_sum::canonicalize() const {
//   std::vector<product_operator> canonical_terms;
//   for (const auto &term : _canonical_terms()) {
//     canonical_terms.push_back(product_operator(term));
//   }
//   return operator_sum(canonical_terms);
// }

// bool operator_sum::operator==(const operator_sum &other) const {
// return _canonical_terms() == other._canonical_terms();
// }

// // Degrees property
// std::vector<int> operator_sum::degrees() const {
//   std::set<int> unique_degrees;
//   for (const auto &term : m_terms) {
//     for (const auto &op : term.get_terms()) {
//       unique_degrees.insert(op.get_degrees().begin(), op.get_degrees().end());
//     }
//   }

//   return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
// }

// // Parameters property
// std::map<std::string, std::string> operator_sum::parameters() const {
//   std::map<std::string, std::string> param_map;
//   for (const auto &term : m_terms) {
//     for (const auto &op : term.get_terms()) {
//       auto op_params = op.parameters();
//       param_map.insert(op_params.begin(), op.params.end());
//     }
//   }

//   return param_map;
// }

// // Check if all terms are spin operators
// bool operator_sum::_is_spinop() const {
//   return std::all_of(
//       m_terms.begin(), m_terms.end(), [](product_operator &term) {
//         return std::all_of(term.get_terms().begin(),
//                            term.get_terms().end(),
//                            [](const Operator &op) { return op.is_spinop();
//                            });
//       });
// }

// Arithmetic operators
operator_sum operator_sum::operator+(const operator_sum &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.insert(combined_terms.end(), other.m_terms.begin(),
                        other.m_terms.end());
  return operator_sum(combined_terms);
}

/// FIXME:
// operator_sum operator_sum::operator-(const operator_sum &other) const {
//   return *this + (-1 * other);
// }

/// FIXME:
// operator_sum operator_sum::operator-=(const operator_sum &other) {
//   *this = *this - other;
//   return *this;
// }

operator_sum operator_sum::operator+=(const operator_sum &other) {
  *this = *this + other;
  return *this;
}

/// FIXME:
// operator_sum operator_sum::operator*(const operator_sum &other) const {
//   std::vector<product_operator> product_terms;
//   for (const auto &self_term : m_terms) {
//     for (const auto &other_term : other.m_terms) {
//       product_terms.push_back(self_term * other_term);
//     }
//   }
// return operator_sum(product_terms);
// }

/// FIXME:
// operator_sum operator_sum::operator/(const operator_sum &other) const {
//   std::vector<product_operator> divided_terms;
//   for (const auto &term : m_terms) {
//     divided_terms.push_back(term / other);
//   }

//   return operator_sum(divided_terms);
// }

// operator_sum + scalar_operator
operator_sum operator_sum::operator+(const scalar_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(product_operator({other}));
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator-(const scalar_operator &other) const {
  return *this + (-1.0 * other);
}

operator_sum operator_sum::operator+=(const scalar_operator &other) {
  *this = *this + other;
  return *this;
}

operator_sum operator_sum::operator-=(const scalar_operator &other) {
  *this = *this - other;
  return *this;
}

// operator_sum + product_operator
operator_sum operator_sum::operator+(const product_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(other);
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator+=(const product_operator &other) {
  *this = *this + other;
  return *this;
}

/// FIXME:
// operator_sum operator_sum::operator-(const product_operator &other) const {
//   return *this + (-1. * other);
// }

/// FIXME:
// operator_sum operator_sum::operator-=(const product_operator &other) {
//   *this = *this - other;
//   return *this;
// }

/// FIXME:
// complex_matrix
// operator_sum::to_matrix(const std::map<int, int> &dimensions,
//                         const std::map<std::string, double> &params) const {
// // todo
// }

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