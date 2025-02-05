/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <map>
#include <vector>
#include "cudaq/operators.h"

namespace cudaq {

class matrix_operator : operator_handler{

private:

  static std::map<std::string, Definition> m_ops;

  std::vector<int> targets;
  std::string id;

public:

  // tools for custom operators

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
  static void define(std::string operator_id, std::vector<int> expected_dimensions,
                     MatrixCallbackFunction &&create);

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  virtual std::vector<int> degrees() const;

  virtual bool is_identity() const;

  // constructors and destructors

  // The constructor should never be called directly by the user:
  // Keeping it internally documented for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  matrix_operator(std::string operator_id, const std::vector<int> &degrees);

  // constructor
  matrix_operator(std::string operator_id, std::vector<int> &&degrees);

  // copy constructor
  matrix_operator(const matrix_operator &other);

  // move constructor
  matrix_operator(matrix_operator &&other);

  ~matrix_operator() = default;

  // assignments

  // assignment operator
  matrix_operator& operator=(const matrix_operator& other);

  // move assignment operator
  matrix_operator& operator=(matrix_operator &&other);

  // evaluations

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2 to_matrix(std::map<int, int> &dimensions,
                             std::map<std::string, std::complex<double>> parameters = {}) const;

  // comparisons

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(const matrix_operator &other) const;

  // predefined operators

  static product_operator<matrix_operator> identity(int degree);
  static product_operator<matrix_operator> annihilate(int degree);
  static product_operator<matrix_operator> create(int degree);
  static product_operator<matrix_operator> momentum(int degree);
  static product_operator<matrix_operator> number(int degree);
  static product_operator<matrix_operator> parity(int degree);
  static product_operator<matrix_operator> position(int degree);
  /// Operators that accept parameters at runtime.
  static product_operator<matrix_operator> squeeze(int degree);
  static product_operator<matrix_operator> displace(int degree);
};

#ifdef CUDAQ_INSTANTIATE_TEMPLATES
template class product_operator<matrix_operator>;
template class operator_sum<matrix_operator>;
#else
extern template class product_operator<matrix_operator>;
extern template class operator_sum<matrix_operator>;
#endif

} // namespace cudaq