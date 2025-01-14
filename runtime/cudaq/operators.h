/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "definition.h"
#include "utils/tensor.h"

#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <variant>

namespace cudaq {

class operator_sum;
class product_operator;
class scalar_operator;
class elementary_operator;
template <typename TEval>
class OperatorArithmetics;
class EvaluatedMatrix;
class MatrixArithmetics;

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
class operator_sum {
  friend class product_operator;
  friend class scalar_operator;
  friend class elementary_operator;

private:
  std::vector<product_operator> m_terms;

  std::vector<std::tuple<scalar_operator, elementary_operator>>
  canonicalize_product(product_operator &prod) const;

  std::vector<std::tuple<scalar_operator, elementary_operator>>
  _canonical_terms() const;

public:
  /// @brief Empty constructor that a user can aggregate terms into.
  operator_sum() = default;

  /// @brief Construct a `cudaq::operator_sum` given a sequence of
  /// `cudaq::product_operator`'s.
  /// This operator expression represents a sum of terms, where each term
  /// is a product of elementary and scalar operators.
  operator_sum(const std::vector<product_operator> &terms);

  operator_sum canonicalize() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  bool _is_spinop() const;

  /// TODO: implement
  // template<typename TEval>
  // TEval _evaluate(OperatorArithmetics<TEval> &arithmetics) const;

  /// @brief Return the `operator_sum` as a matrix.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the paramter names to their concrete, complex
  /// values.
  matrix_2 to_matrix(
      const std::map<int, int> &dimensions,
      const std::map<std::string, std::complex<double>> &params = {}) const;

  // Arithmetic operators
  operator_sum operator+(const operator_sum &other) const;
  operator_sum operator-(const operator_sum &other) const;
  operator_sum operator*(operator_sum &other) const;
  operator_sum operator*=(operator_sum &other);
  operator_sum operator+=(const operator_sum &other);
  operator_sum operator-=(const operator_sum &other);
  operator_sum operator*(const scalar_operator &other) const;
  operator_sum operator+(const scalar_operator &other) const;
  operator_sum operator-(const scalar_operator &other) const;
  operator_sum operator*=(const scalar_operator &other);
  operator_sum operator+=(const scalar_operator &other);
  operator_sum operator-=(const scalar_operator &other);
  operator_sum operator*(std::complex<double> other) const;
  operator_sum operator+(std::complex<double> other) const;
  operator_sum operator-(std::complex<double> other) const;
  operator_sum operator*=(std::complex<double> other);
  operator_sum operator+=(std::complex<double> other);
  operator_sum operator-=(std::complex<double> other);
  operator_sum operator*(double other) const;
  operator_sum operator+(double other) const;
  operator_sum operator-(double other) const;
  operator_sum operator*=(double other);
  operator_sum operator+=(double other);
  operator_sum operator-=(double other);
  operator_sum operator*(const product_operator &other) const;
  operator_sum operator+(const product_operator &other) const;
  operator_sum operator-(const product_operator &other) const;
  operator_sum operator*=(const product_operator &other);
  operator_sum operator+=(const product_operator &other);
  operator_sum operator-=(const product_operator &other);
  operator_sum operator+(const elementary_operator &other) const;
  operator_sum operator-(const elementary_operator &other) const;
  operator_sum operator*(const elementary_operator &other) const;
  operator_sum operator*=(const elementary_operator &other);
  operator_sum operator+=(const elementary_operator &other);
  operator_sum operator-=(const elementary_operator &other);
  friend operator_sum operator*(std::complex<double> other, operator_sum self);
  friend operator_sum operator+(std::complex<double> other, operator_sum self);
  friend operator_sum operator-(std::complex<double> other, operator_sum self);
  friend operator_sum operator*(double other, operator_sum self);
  friend operator_sum operator+(double other, operator_sum self);
  friend operator_sum operator-(double other, operator_sum self);

  /// @brief Return the operator_sum as a string.
  std::string to_string() const;

  /// @brief Return the number of operator terms that make up this operator sum.
  int term_count() const { return m_terms.size(); }

  /// @brief  True, if the other value is an operator_sum with equivalent terms,
  /// and False otherwise. The equality takes into account that operator
  /// addition is commutative, as is the product of two operators if they
  /// act on different degrees of freedom.
  /// The equality comparison does *not* take commutation relations into
  /// account, and does not try to reorder terms blockwise; it may hence
  /// evaluate to False, even if two operators in reality are the same.
  /// If the equality evaluates to True, on the other hand, the operators
  /// are guaranteed to represent the same transformation for all arguments.
  bool operator==(const operator_sum &other) const;
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
class product_operator : public operator_sum {
  friend class scalar_operator;
  friend class elementary_operator;

private:
  std::vector<std::variant<scalar_operator, elementary_operator>> m_terms;
  std::vector<scalar_operator> m_scalar_ops;
  std::vector<elementary_operator> m_elementary_ops;
  cudaq::matrix_2
  m_evaluate(MatrixArithmetics arithmetics, std::map<int, int> dimensions,
             std::map<std::string, std::complex<double>> parameters,
             bool pad_terms = true);

public:
  product_operator() = default;
  ~product_operator() = default;

  // Constructor for an operator expression that represents a product
  // of scalar and elementary operators.
  // arg atomic_operators : The operators of which to compute the product when
  //                         evaluating the operator expression.
  product_operator(std::vector<scalar_operator> scalars,
                   std::vector<elementary_operator> atomic_operators);

  product_operator(
      std::vector<std::variant<scalar_operator, elementary_operator>>
          atomic_operators);

  // Arithmetic overloads against all other operator types.
  operator_sum operator+(std::complex<double> other);
  operator_sum operator-(std::complex<double> other);
  product_operator operator*(std::complex<double> other);
  product_operator operator*=(std::complex<double> other);
  operator_sum operator+(double other);
  operator_sum operator-(double other);
  product_operator operator*(double other);
  product_operator operator*=(double other);
  operator_sum operator+(scalar_operator other);
  operator_sum operator-(scalar_operator other);
  product_operator operator*(scalar_operator other);
  product_operator operator*=(scalar_operator other);
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  product_operator operator*(product_operator other);
  product_operator operator*=(product_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  product_operator operator*(elementary_operator other);
  product_operator operator*=(elementary_operator other);
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator*(operator_sum other);

  friend operator_sum operator+(std::complex<double> other,
                                product_operator self);
  friend operator_sum operator-(std::complex<double> other,
                                product_operator self);
  friend product_operator operator*(std::complex<double> other,
                                    product_operator self);
  friend operator_sum operator+(double other, product_operator self);
  friend operator_sum operator-(double other, product_operator self);
  friend product_operator operator*(double other, product_operator self);

  /// @brief True, if the other value is an operator_sum with equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms blockwise; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(product_operator other);

  /// @brief Return the `product_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the paramter names to their concrete, complex
  /// values.
  matrix_2 to_matrix(std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters);

  /// @brief Creates a representation of the operator as a `cudaq::pauli_word`
  /// that can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word();

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  int term_count() const { return m_terms.size(); }
};

class elementary_operator : public product_operator {
private:
  std::map<std::string, Definition> m_ops;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documentd for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  elementary_operator(std::string operator_id, std::vector<int> degrees);

  // Copy constructor.
  elementary_operator(const elementary_operator &other);
  elementary_operator(elementary_operator &other);

  // Arithmetic overloads against all other operator types.
  operator_sum operator+(std::complex<double> other);
  operator_sum operator-(std::complex<double> other);
  product_operator operator*(std::complex<double> other);
  operator_sum operator+(double other);
  operator_sum operator-(double other);
  product_operator operator*(double other);
  operator_sum operator+(scalar_operator other);
  operator_sum operator-(scalar_operator other);
  product_operator operator*(scalar_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  product_operator operator*(elementary_operator other);
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  product_operator operator*(product_operator other);
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator+=(operator_sum other);
  operator_sum operator-=(operator_sum other);
  operator_sum operator*(operator_sum other);

  // Reverse order arithmetic for elementary operators against pure scalars.
  friend operator_sum operator+(std::complex<double> other,
                                elementary_operator self);
  friend operator_sum operator-(std::complex<double> other,
                                elementary_operator self);
  friend product_operator operator*(std::complex<double> other,
                                    elementary_operator self);
  friend operator_sum operator+(double other, elementary_operator self);
  friend operator_sum operator-(double other, elementary_operator self);
  friend product_operator operator*(double other, elementary_operator self);

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(elementary_operator other);

  /// @brief Return the `elementary_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `elementary_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  matrix_2 to_matrix(std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters);

  // Predefined operators.
  static elementary_operator identity(int degree);
  static elementary_operator zero(int degree);
  static elementary_operator annihilate(int degree);
  static elementary_operator create(int degree);
  static elementary_operator momentum(int degree);
  static elementary_operator number(int degree);
  static elementary_operator parity(int degree);
  static elementary_operator position(int degree);
  /// Operators that require runtime parameters from the user.
  static elementary_operator squeeze(int degree);
  static elementary_operator displace(int degree);

  /// @brief Adds the definition of an elementary operator with the given id to
  /// the class. After definition, an the defined elementary operator can be
  /// instantiated by providing the operator id as well as the degree(s) of
  /// freedom that it acts on. An elementary operator is a parameterized object
  /// acting on certain degrees of freedom. To evaluate an operator, for example
  /// to compute its matrix, the level, that is the dimension, for each degree
  /// of freedom it acts on must be provided, as well as all additional
  /// parameters. Additional parameters must be provided in the form of keyword
  /// arguments. Note: The dimensions passed during operator evaluation are
  /// automatically validated against the expected dimensions specified during
  /// definition - the `create` function does not need to do this.
  /// @arg operator_id : A string that uniquely identifies the defined operator.
  /// @arg expected_dimensions : Defines the number of levels, that is the
  /// dimension,
  ///      for each degree of freedom in canonical (that is sorted) order. A
  ///      negative or zero value for one (or more) of the expected dimensions
  ///      indicates that the operator is defined for any dimension of the
  ///      corresponding degree of freedom.
  /// @arg create : Takes any number of complex-valued arguments and returns the
  ///      matrix representing the operator in canonical order. If the matrix
  ///      can be defined for any number of levels for one or more degree of
  ///      freedom, the `create` function must take an argument called
  ///      `dimension` (or `dim` for short), if the operator acts on a single
  ///      degree of freedom, and an argument called `dimensions` (or `dims` for
  ///      short), if the operator acts
  ///     on multiple degrees of freedom.
  template <typename Func>
  void define(std::string operator_id, std::map<int, int> expected_dimensions,
              Func create) {
    if (m_ops.find(operator_id) != m_ops.end()) {
      // todo: make a nice error message to say op already exists
      throw;
    }
    auto defn = Definition();
    defn.create_definition(operator_id, expected_dimensions, create);
    m_ops[operator_id] = defn;
  }

  // Attributes.

  /// @brief The number of levels, that is the dimension, for each degree of
  /// freedom in canonical order that the operator acts on. A value of zero or
  /// less indicates that the operator is defined for any dimension of that
  /// degree.
  std::map<int, int> expected_dimensions;
  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;
  std::string id;

  // /// @brief Creates a representation of the operator as `pauli_word` that
  // can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word ovveride();
};

class scalar_operator : public product_operator {
private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::complex<double> m_constant_value;

  // Only populated when we've performed arithmetic between various
  // scalar operators.
  std::vector<scalar_operator> m_operators_to_compose;

  /// @brief The function that generates the value of the scalar operator.
  /// The function can take a vector of complex-valued arguments
  /// and returns a number.
  ScalarCallbackFunction m_generator;

public:
  /// @brief Constructor that just takes a callback function with no
  /// arguments.
  scalar_operator(ScalarCallbackFunction &&create) {
    m_generator = ScalarCallbackFunction(create);
  }

  /// @brief Constructor that just takes and returns a complex double value.
  /// @NOTE: This replicates the behavior of the python `scalar_operator::const`
  /// without the need for an extra member function.
  scalar_operator(std::complex<double> value);
  scalar_operator(double value);

  // Arithmetic overloads against other operator types.
  scalar_operator operator+(scalar_operator other);
  scalar_operator operator-(scalar_operator other);
  scalar_operator operator*(scalar_operator other);
  scalar_operator operator/(scalar_operator other);
  /// TODO: implement and test pow
  scalar_operator pow(scalar_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  product_operator operator*(elementary_operator other);
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  product_operator operator*(product_operator other);
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator*(operator_sum other);
  friend scalar_operator operator+(scalar_operator self,
                                   std::complex<double> other);
  friend scalar_operator operator-(scalar_operator self,
                                   std::complex<double> other);
  friend scalar_operator operator*(scalar_operator self,
                                   std::complex<double> other);
  friend scalar_operator operator/(scalar_operator self,
                                   std::complex<double> other);
  friend scalar_operator operator+(std::complex<double> other,
                                   scalar_operator self);
  friend scalar_operator operator-(std::complex<double> other,
                                   scalar_operator self);
  friend scalar_operator operator*(std::complex<double> other,
                                   scalar_operator self);
  friend scalar_operator operator/(std::complex<double> other,
                                   scalar_operator self);
  friend scalar_operator operator+(scalar_operator self, double other);
  friend scalar_operator operator-(scalar_operator self, double other);
  friend scalar_operator operator*(scalar_operator self, double other);
  friend scalar_operator operator/(scalar_operator self, double other);
  friend scalar_operator operator+(double other, scalar_operator self);
  friend scalar_operator operator-(double other, scalar_operator self);
  friend scalar_operator operator*(double other, scalar_operator self);
  friend scalar_operator operator/(double other, scalar_operator self);
  friend void operator+=(scalar_operator &self, std::complex<double> other);
  friend void operator-=(scalar_operator &self, std::complex<double> other);
  friend void operator*=(scalar_operator &self, std::complex<double> other);
  friend void operator/=(scalar_operator &self, std::complex<double> other);
  friend void operator+=(scalar_operator &self, scalar_operator other);
  friend void operator-=(scalar_operator &self, scalar_operator other);
  friend void operator*=(scalar_operator &self, scalar_operator other);
  friend void operator/=(scalar_operator &self, scalar_operator other);

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(std::map<std::string, std::complex<double>> parameters);

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatability with the other inherited classes.
  matrix_2 to_matrix(std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters);

  // /// @brief Returns true if other is a scalar operator with the same
  // /// generator.
  // bool operator==(scalar_operator other);

  scalar_operator() = default;
  // Copy constructor.
  scalar_operator(const scalar_operator &other);
  scalar_operator(scalar_operator &other);
  ~scalar_operator() = default;

  // Need this property for consistency with other inherited types.
  // Particularly, to be used when the scalar operator is held within
  // a variant type next to elementary operators.
  std::vector<int> degrees = {-1};
};

template <typename TEval>
class OperatorArithmetics {
public:
  /// @brief Accesses the relevant data to evaluate an operator expression
  /// in the leaf nodes, that is in elementary and scalar operators.
  TEval evaluate(std::variant<scalar_operator, elementary_operator> op);

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
  EvaluatedMatrix() = default;
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
  evaluate(std::variant<scalar_operator, elementary_operator> op);
};

} // namespace cudaq