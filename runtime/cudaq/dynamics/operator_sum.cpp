/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "cudaq/operators.h"

#include <iostream>
#include <set>
#include <concepts>
#include <type_traits>

namespace cudaq {

// private methods

template<typename HandlerTy>
std::vector<std::tuple<scalar_operator, HandlerTy>> operator_sum<HandlerTy>::canonicalize_product(product_operator<HandlerTy> &prod) const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
std::vector<std::tuple<scalar_operator, HandlerTy>> operator_sum<HandlerTy>::_canonical_terms() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
void operator_sum<HandlerTy>::aggregate_terms() {}

template<typename HandlerTy>
template <typename ... Args>
void operator_sum<HandlerTy>::aggregate_terms(const product_operator<HandlerTy> &head, Args&& ... args) {
    this->terms.push_back(head.terms[0]);
    this->coefficients.push_back(head.coefficients[0]);
    aggregate_terms(std::forward<Args>(args)...);
}

template
std::vector<std::tuple<scalar_operator, elementary_operator>> operator_sum<elementary_operator>::canonicalize_product(product_operator<elementary_operator> &prod) const;

template
std::vector<std::tuple<scalar_operator, elementary_operator>> operator_sum<elementary_operator>::_canonical_terms() const;

// no overload for a single product, since we don't want a constructor for a single term

template
void operator_sum<elementary_operator>::aggregate_terms(const product_operator<elementary_operator> &item1, 
                                                        const product_operator<elementary_operator> &item2);

template
void operator_sum<elementary_operator>::aggregate_terms(const product_operator<elementary_operator> &item1, 
                                                        const product_operator<elementary_operator> &item2,
                                                        const product_operator<elementary_operator> &item3);

// read-only properties

template<typename HandlerTy>
std::vector<int> operator_sum<HandlerTy>::degrees() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
int operator_sum<HandlerTy>::n_terms() const { 
    return this->terms.size(); 
}

template<typename HandlerTy>
std::vector<product_operator<HandlerTy>> operator_sum<HandlerTy>::get_terms() const { 
    std::vector<product_operator<HandlerTy>> prods;
    prods.reserve(this->terms.size());
    for (size_t i = 0; i < this->terms.size(); ++i) {
        prods.push_back(product_operator<HandlerTy>(this->coefficients[i], this->terms[i]));
    }
    return prods; 
}

template
std::vector<int> operator_sum<elementary_operator>::degrees() const;

template
int operator_sum<elementary_operator>::n_terms() const;

template
std::vector<product_operator<elementary_operator>> operator_sum<elementary_operator>::get_terms() const;

// constructors

template<typename HandlerTy>
template<class... Args, class>
operator_sum<HandlerTy>::operator_sum(const Args&... args) {
    this->terms.reserve(sizeof...(Args));
    this->coefficients.reserve(sizeof...(Args));
    aggregate_terms(args...);
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const std::vector<product_operator<HandlerTy>> &terms) { 
    this->terms.reserve(terms.size());
    this->coefficients.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
        this->terms.push_back(term.terms[0]);
        this->coefficients.push_back(term.coefficients[0]);
    }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(std::vector<product_operator<HandlerTy>> &&terms) { 
    this->terms.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
        this->terms.push_back(std::move(term.terms[0]));
        this->coefficients.push_back(std::move(term.coefficients[0]));
    }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other)
    : coefficients(other.coefficients), terms(other.terms) {}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other) 
    : coefficients(std::move(other.coefficients)), terms(std::move(other.terms)) {}

// no constructor for a single product, since that one should remain a product op

template 
operator_sum<elementary_operator>::operator_sum(const product_operator<elementary_operator> &item1,
                                                const product_operator<elementary_operator> &item2);

template 
operator_sum<elementary_operator>::operator_sum(const product_operator<elementary_operator> &item1,
                                                const product_operator<elementary_operator> &item2,
                                                const product_operator<elementary_operator> &item3);

template
operator_sum<elementary_operator>::operator_sum(const std::vector<product_operator<elementary_operator>> &terms);

template
operator_sum<elementary_operator>::operator_sum(std::vector<product_operator<elementary_operator>> &&terms);

template
operator_sum<elementary_operator>::operator_sum(const operator_sum<elementary_operator> &other);

template
operator_sum<elementary_operator>::operator_sum(operator_sum<elementary_operator> &&other);

// assignments

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const operator_sum<HandlerTy> &other) {
    if (this != &other) {
        coefficients = other.coefficients;
        terms = other.terms;
    }
    return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(operator_sum<HandlerTy> &&other) {
    if (this != &other) {
        coefficients = std::move(other.coefficients);
        terms = std::move(other.terms);
    }
    return *this;
}

template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator=(const operator_sum<elementary_operator>& other);

template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator=(operator_sum<elementary_operator> &&other);

// evaluations

template<typename HandlerTy>
std::string operator_sum<HandlerTy>::to_string() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
matrix_2 operator_sum<HandlerTy>::to_matrix(const std::map<int, int> &dimensions,
                                            const std::map<std::string, double> &params) const {
    throw std::runtime_error("not implemented");
}

template
std::string operator_sum<elementary_operator>::to_string() const;

template
matrix_2 operator_sum<elementary_operator>::to_matrix(const std::map<int, int> &dimensions,
                                                      const std::map<std::string, double> &params) const;

// comparisons

template<typename HandlerTy>
bool operator_sum<HandlerTy>::operator==(const operator_sum<HandlerTy> &other) const {
    throw std::runtime_error("not implemented");
}

template
bool operator_sum<elementary_operator>::operator==(const operator_sum<elementary_operator> &other) const;

// unary operators

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() const {
  std::vector<scalar_operator> coefficients;
  coefficients.reserve(this->coefficients.size());
  for (auto coeff : this->coefficients)
    coefficients.push_back(-1. * coeff);
  operator_sum<HandlerTy> sum;
  sum.coefficients = std::move(coefficients);
  sum.terms = this->terms;
  return sum;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() const {
  return *this;
}

template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-() const;

template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+() const;

// left-hand arithmetics

#define SUM_MULTIPLICATION_REVERSE(otherTy)                                             \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator*(otherTy other,                                      \
                                    const operator_sum<HandlerTy> &self) {              \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(self.coefficients.size());                                     \
    for (auto coeff : self.coefficients)                                                \
      coefficients.push_back(coeff * other);                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = self.terms;                                                             \
    return sum;                                                                         \
  }

SUM_MULTIPLICATION_REVERSE(double);
SUM_MULTIPLICATION_REVERSE(std::complex<double>);
SUM_MULTIPLICATION_REVERSE(const scalar_operator &);

#define SUM_ADDITION_REVERSE(otherTy, op)                                               \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      const operator_sum<HandlerTy> &self) {            \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(self.terms.size() + 1);                                        \
    coefficients.push_back(other);                                                      \
    for (auto coeff : self.coefficients)                                                \
      coefficients.push_back(op coeff);                                                 \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(self.terms.size() + 1);                                               \
    terms.push_back({});                                                                \
    for (auto term : self.terms)                                                        \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_REVERSE(double, +);
SUM_ADDITION_REVERSE(double, -);
SUM_ADDITION_REVERSE(std::complex<double>, +);
SUM_ADDITION_REVERSE(std::complex<double>, -);
SUM_ADDITION_REVERSE(const scalar_operator &, +);
SUM_ADDITION_REVERSE(const scalar_operator &, -);

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const HandlerTy &other, const operator_sum<HandlerTy> &self) {
  std::vector<std::vector<HandlerTy>> terms;
  terms.reserve(self.terms.size());
  for (auto term : self.terms) {
    std::vector<HandlerTy> prod;
    prod.reserve(term.size() + 1);
    prod.push_back(other);
    for (auto op : term)
      prod.push_back(op);
    terms.push_back(std::move(prod));
  }
  operator_sum<HandlerTy> sum;
  sum.coefficients = self.coefficients;
  sum.terms = std::move(terms);
  return sum;
}

#define SUM_ADDITION_HANDLER_REVERSE(op)                                                \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(const HandlerTy & other,                          \
                                      const operator_sum<HandlerTy> &self) {            \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(self.terms.size() + 1);                                        \
    coefficients.push_back(1.);                                                         \
    for (auto coeff : self.coefficients)                                                \
      coefficients.push_back(op coeff);                                                 \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(self.terms.size() + 1);                                               \
    std::vector<HandlerTy> newTerm;                                                     \
    newTerm.push_back(other);                                                           \
    terms.push_back(std::move(newTerm));                                                \
    for (auto term : self.terms)                                                        \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_HANDLER_REVERSE(+)
SUM_ADDITION_HANDLER_REVERSE(-)

template
operator_sum<elementary_operator> operator*(const scalar_operator &other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator*(std::complex<double> other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator*(double other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator*(const elementary_operator &other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(const scalar_operator &other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(double other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(std::complex<double> other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(const elementary_operator &other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(const scalar_operator &other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(double other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(std::complex<double> other, const operator_sum<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(const elementary_operator &other, const operator_sum<elementary_operator> &self);

} // namespace cudaq