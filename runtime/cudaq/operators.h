/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <vector>
#include <unordered_map>
#include <type_traits>

#include "utils/tensor.h"
#include "dynamics/operator_leafs.h"
#include "dynamics/templates.h"

namespace cudaq {

class MatrixArithmetics;
class EvaluatedMatrix;

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy>
class operator_sum {
template <typename T> friend class operator_sum;
template <typename T> friend class product_operator;

private:

  // inserts a new term combining it with an existing one if possible
  void insert(product_operator<HandlerTy> &&other);
  void insert(const product_operator<HandlerTy> &other);

  void aggregate_terms();

  template <typename ... Args>
  void aggregate_terms(product_operator<HandlerTy> &&head, Args&& ... args);

  EvaluatedMatrix m_evaluate(MatrixArithmetics arithmetics, bool pad_terms = true) const;

protected:

  std::unordered_map<std::string, int> term_map; // quick access to term index given its id (used for aggregating terms)
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

  template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<product_operator<HandlerTy>, Args>...>::value, bool> = true>
  operator_sum(Args&&... args);

public:

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this operator sum.
  int num_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  std::vector<product_operator<HandlerTy>> get_terms() const;

  // constructors and destructors

  operator_sum(const product_operator<HandlerTy> &other);

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  operator_sum(const operator_sum<T> &other);

  // copy constructor
  operator_sum(const operator_sum<HandlerTy> &other, int size = 0);

  // move constructor
  operator_sum(operator_sum<HandlerTy> &&other, int size = 0);

  ~operator_sum() = default;

  // assignments

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  operator_sum<HandlerTy>& operator=(const product_operator<T> &other);

  operator_sum<HandlerTy>& operator=(const product_operator<HandlerTy> &other);

  operator_sum<HandlerTy>& operator=(product_operator<HandlerTy> &&other);

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  operator_sum<HandlerTy>& operator=(const operator_sum<T> &other);

  // assignment operator
  operator_sum<HandlerTy>& operator=(const operator_sum<HandlerTy> &other);

  // move assignment operator
  operator_sum<HandlerTy>& operator=(operator_sum<HandlerTy> &&other);

  // evaluations

  /// @brief Return the operator_sum<HandlerTy> as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum<HandlerTy>` as a matrix.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(std::unordered_map<int, int> dimensions = {},
                     const std::unordered_map<std::string, std::complex<double>> &parameters = {}) const;

  // unary operators

  operator_sum<HandlerTy> operator-() const &;
  operator_sum<HandlerTy> operator-() &&;
  operator_sum<HandlerTy> operator+() const &;
  operator_sum<HandlerTy> operator+() &&;

  // right-hand arithmetics

  operator_sum<HandlerTy> operator*(double other) const &;
  operator_sum<HandlerTy> operator*(double other) &&;
  operator_sum<HandlerTy> operator+(double other) const &;
  operator_sum<HandlerTy> operator+(double other) &&;
  operator_sum<HandlerTy> operator-(double other) const &;
  operator_sum<HandlerTy> operator-(double other) &&;
  operator_sum<HandlerTy> operator*(std::complex<double> other) const &;
  operator_sum<HandlerTy> operator*(std::complex<double> other) &&;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const &;
  operator_sum<HandlerTy> operator+(std::complex<double> other) &&;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const &;
  operator_sum<HandlerTy> operator-(std::complex<double> other) &&;
  operator_sum<HandlerTy> operator*(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator*(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) &&;

  operator_sum<HandlerTy>& operator*=(double other);
  operator_sum<HandlerTy>& operator+=(double other);
  operator_sum<HandlerTy>& operator-=(double other);
  operator_sum<HandlerTy>& operator*=(std::complex<double> other);
  operator_sum<HandlerTy>& operator+=(std::complex<double> other);
  operator_sum<HandlerTy>& operator-=(std::complex<double> other);
  operator_sum<HandlerTy>& operator*=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator+=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator-=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(product_operator<HandlerTy> &&other);
  operator_sum<HandlerTy>& operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator-=(product_operator<HandlerTy> &&other);
  operator_sum<HandlerTy>& operator*=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(operator_sum<HandlerTy> &&other);
  operator_sum<HandlerTy>& operator-=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator-=(operator_sum<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template instantiation is a nightmare.
  template<typename T>
  friend operator_sum<T> operator*(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(double other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(double other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator*(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(std::complex<double> other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator*(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(const scalar_operator &other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, operator_sum<T> &&self);

  template<typename T>
  friend operator_sum<T> operator+(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(double other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, product_operator<T> &&self);

  // common operators

  template<typename T>
  friend operator_sum<T> operator_handler::empty();
};


/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy>
class product_operator {
template <typename T> friend class product_operator;
template <typename T> friend class operator_sum;

private:
  // template defined as long as T implements an in-place multiplication -
  // won't work if the in-place multiplication was inherited from a base class
  template <typename T>
  static decltype(std::declval<T>().inplace_mult(std::declval<T>())) handler_mult(int);
	template<typename T> 
  static std::false_type handler_mult(...); // ellipsis ensures the template above is picked if it exists
	static constexpr bool supports_inplace_mult = !std::is_same<decltype(handler_mult<HandlerTy>(0)), std::false_type>::value;

#if !defined(NDEBUG)
  bool is_canonicalized() const;
#endif
 
  typename std::vector<HandlerTy>::const_iterator find_insert_at(const HandlerTy &other) const;

  template<typename T, std::enable_if_t<std::is_same<HandlerTy, T>::value && 
                                        !product_operator<T>::supports_inplace_mult, 
                       std::false_type> = std::false_type()>
  void insert(T &&other);

  template<typename T, std::enable_if_t<std::is_same<HandlerTy, T>::value && 
                                        product_operator<T>::supports_inplace_mult,
                       std::true_type> = std::true_type()>
  void insert(T &&other);

  std::string get_term_id() const;

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(HandlerTy &&head, Args&& ... args);

  EvaluatedMatrix m_evaluate(MatrixArithmetics arithmetics, bool pad_terms = true) const;

protected:

  std::vector<HandlerTy> operators;
  scalar_operator coefficient;

  template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<HandlerTy, Args>...>::value, bool> = true>
  product_operator(scalar_operator coefficient, Args&&... args);

  // keep this constructor protected (otherwise it needs to ensure canonical order)
  product_operator(scalar_operator coefficient, const std::vector<HandlerTy> &atomic_operators, int size = 0);

  // keep this constructor protected (otherwise it needs to ensure canonical order)
  product_operator(scalar_operator coefficient, std::vector<HandlerTy> &&atomic_operators, int size = 0);

public:

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  int num_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  const std::vector<HandlerTy>& get_terms() const;

  scalar_operator get_coefficient() const;

  // constructors and destructors

  product_operator(double coefficient);

  product_operator(HandlerTy &&atomic);

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  product_operator(const product_operator<T> &other);

  // copy constructor
  product_operator(const product_operator<HandlerTy> &other, int size = 0);

  // move constructor
  product_operator(product_operator<HandlerTy> &&other, int size = 0);

  ~product_operator() = default;

  // assignments

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  product_operator<HandlerTy>& operator=(const product_operator<T> &other);

  // assignment operator
  product_operator<HandlerTy>& operator=(const product_operator<HandlerTy> &other);

  // move assignment operator
  product_operator<HandlerTy>& operator=(product_operator<HandlerTy> &&other);

  // evaluations

  /// @brief Return the `product_operator<HandlerTy>` as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum<HandlerTy>` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(std::unordered_map<int, int> dimensions = {},
                     const std::unordered_map<std::string, std::complex<double>> &parameters = {}) const;

  // comparisons

  /// @brief True, if the other value is an operator_sum<HandlerTy> with equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms `blockwise`; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(const product_operator<HandlerTy> &other) const;

  // unary operators

  product_operator<HandlerTy> operator-() const &;
  product_operator<HandlerTy> operator-() &&;
  product_operator<HandlerTy> operator+() const &;
  product_operator<HandlerTy> operator+() &&;

  // right-hand arithmetics

  product_operator<HandlerTy> operator*(double other) const &;
  product_operator<HandlerTy> operator*(double other) &&;
  operator_sum<HandlerTy> operator+(double other) const &;
  operator_sum<HandlerTy> operator+(double other) &&;
  operator_sum<HandlerTy> operator-(double other) const &;
  operator_sum<HandlerTy> operator-(double other) &&;
  product_operator<HandlerTy> operator*(std::complex<double> other) const &;
  product_operator<HandlerTy> operator*(std::complex<double> other) &&;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const &;
  operator_sum<HandlerTy> operator+(std::complex<double> other) &&;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const &;
  operator_sum<HandlerTy> operator-(std::complex<double> other) &&;
  product_operator<HandlerTy> operator*(const scalar_operator &other) const &;
  product_operator<HandlerTy> operator*(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) &&;
  product_operator<HandlerTy> operator*(const product_operator<HandlerTy> &other) const &;
  product_operator<HandlerTy> operator*(const product_operator<HandlerTy> &other) &&;
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> &&other) const &;
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) &&;

  product_operator<HandlerTy>& operator*=(double other);
  product_operator<HandlerTy>& operator*=(std::complex<double> other);
  product_operator<HandlerTy>& operator*=(const scalar_operator &other);
  product_operator<HandlerTy>& operator*=(const product_operator<HandlerTy> &other);
  product_operator<HandlerTy>& operator*=(product_operator<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template instantiation is a nightmare.
  template<typename T>
  friend product_operator<T> operator*(double other, const product_operator<T> &self);
  template<typename T>
  friend product_operator<T> operator*(double other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(double other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, product_operator<T> &&self);
  template<typename T>
  friend product_operator<T> operator*(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend product_operator<T> operator*(std::complex<double> other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, product_operator<T> &&self);
  template<typename T>
  friend product_operator<T> operator*(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend product_operator<T> operator*(const scalar_operator &other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, product_operator<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, product_operator<T> &&self);

  template<typename T>
  friend operator_sum<T> operator*(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(double other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(double other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator*(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(std::complex<double> other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator*(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(const scalar_operator &other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, operator_sum<T> &&self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, operator_sum<T> &&self);

  // common operators

  template<typename T, typename... Args, std::enable_if_t<std::conjunction<std::is_same<int, Args>...>::value, bool>>
  friend product_operator<T> operator_handler::identity(Args... targets);
};

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
extern template class product_operator<matrix_operator>;
extern template class product_operator<spin_operator>;
extern template class product_operator<boson_operator>;

extern template class operator_sum<matrix_operator>;
extern template class operator_sum<spin_operator>;
extern template class operator_sum<boson_operator>;
#endif

} // namespace cudaq
