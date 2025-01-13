/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

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
    : m_terms(atomic_operators) {}

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

  // // the degrees of freedom this product operator acts upon
  // auto degrees = this->degrees();

  // // the degrees of freedom from the user
  // auto keys = std::views::keys(dimensions);
  // std::vector<int> provided_degrees{keys.begin(), keys.end()};

  // auto accumulate_ops = [&](){
  //   std::vector<elementary_operator> ops;
  //   for (auto degree : provided_degrees) {
  //     if (std::find(degrees.begin(), degrees.end(), degree) == degrees.end())
  //     {
  //       ops.push_back(elementary_operator::identity(degree))
  //     }
  //   }
  // }

  // // are there any extra degrees from the user ?

  // // are there any degrees from the user missing here?

  // // insert

  // // Loop through the terms of the product operator and cast each
  // // to the full Hilbert Space.
  // for (auto &term : m_terms) {

  // }
  return cudaq::matrix_2();
}
} // namespace cudaq