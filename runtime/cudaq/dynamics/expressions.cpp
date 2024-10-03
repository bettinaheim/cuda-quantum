/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/expressions.h"
#include "common/EigenDense.h"

#include <iostream>

namespace cudaq {

ElementaryOperator::ElementaryOperator(std::string operator_id,
                                       std::vector<int> degrees)
    : id(operator_id), degrees(degrees) {}

ElementaryOperator ElementaryOperator::identity(int degree) {
  std::string op_id = "identity";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      int degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      // Build up the identity matrix.
      for (std::size_t i = 0; i < dimension; i++) {
        mat(i, i) = 1.0 + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::zero(int degree) {
  std::string op_id = "zero";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      mat.set_zero();
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::annihilate(int degree) {
  std::string op_id = "annihilate";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i, i + 1) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::create(int degree) {
  std::string op_id = "create";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i + 1, i) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::position(int degree) {
  std::string op_id = "position";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = cudaq::complex_matrix(dimension, dimension);
      // position = 0.5 * (create + annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i + 1, i) = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat(i, i + 1) = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::momentum(int degree) {
  std::string op_id = "momentum";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = cudaq::complex_matrix(dimension, dimension);
      // momentum = 0.5j * (create - annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i + 1, i) =
            (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat(i, i + 1) =
            -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::number(int degree) {
  std::string op_id = "number";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat(i, i) = static_cast<double>(i) + 0.0j;
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::parity(int degree) {
  std::string op_id = "parity";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat(i, i) = std::pow(-1., static_cast<double>(i)) + 0.0j;
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

ElementaryOperator
ElementaryOperator::displace(int degree, std::complex<double> amplitude) {
  std::string op_id = "displace";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto temp_mat = cudaq::complex_matrix(dimension, dimension);
      // displace = exp[ (amplitude * create) - (conj(amplitude) * annihilate) ]
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        temp_mat(i + 1, i) =
            amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        temp_mat(i, i + 1) =
            -1. * std::conj(amplitude) * std::sqrt(static_cast<double>(i + 1)) +
            0.0 * 'j';
      }
      // Not ideal that our method of computing the matrix exponential
      // requires copies here. Maybe we can just use eigen directly here
      // to limit to one copy, but we can address that later.
      auto mat = temp_mat.exp();
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  throw std::runtime_error("currently have a bug in implementation.");
  return op;
}

ElementaryOperator ElementaryOperator::squeeze(int degree,
                                               std::complex<double> amplitude) {
  throw std::runtime_error("Not yet implemented.");
}

complex_matrix ElementaryOperator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {
  std::variant<complex_matrix, std::complex<double>> result =
      m_ops[id].generator(dimensions, parameters);

  if (std::holds_alternative<complex_matrix>(result)) {
    // Move the complex_matrix from the variant, which avoids copying
    return std::move(std::get<complex_matrix>(result));
  } else {
    // if it's a scalar, convert the scalar to a 1x1 matrix
    std::complex<double> scalar = std::get<std::complex<double>>(result);

    cudaq::complex_matrix scalar_matrix(1, 1);
    scalar_matrix(0, 0) = scalar;

    return scalar_matrix;
  }
}

ScalarOperator::ScalarOperator(const ScalarOperator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}
ScalarOperator::ScalarOperator(ScalarOperator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}

ScalarOperator &ScalarOperator::operator=(ScalarOperator &other) {
  generator = other.generator;
  m_constant_value = other.m_constant_value;
  return *this;
}

/// @brief Constructor that just takes and returns a complex double value.
ScalarOperator::ScalarOperator(std::complex<double> value) {
  m_constant_value = value;
  auto func = [&](std::map<std::string, std::complex<double>> _none) {
    return m_constant_value;
  };
  generator = scalar_callback_function(func);
}

std::complex<double> ScalarOperator::evaluate(
    std::map<std::string, std::complex<double>> parameters) {
  /// TODO: throw an error if someone is supposed to call the merged
  // generator instead of this one.
  return generator(parameters);
}

#define ARITHMETIC_OPERATIONS_DOUBLES(op)                                      \
  ScalarOperator operator op(std::complex<double> other,                       \
                             ScalarOperator self) {                            \
    /* Create an operator for the complex double value. */                     \
    auto otherOperator = ScalarOperator(other);                                \
    /* Create an operator that we will store the result in and return to the   \
     * user. */                                                                \
    ScalarOperator returnOperator;                                             \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    returnOperator._operators_to_compose.push_back(self);                      \
    returnOperator._operators_to_compose.push_back(otherOperator);             \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          /* FIXME: I have to use this hacky `.get_val()` on the newly created \
           * operator for the given complex double -- because calling the      \
           * evaluate function returns 0.0 . I have no clue why??? */          \
          return returnOperator._operators_to_compose[0]                       \
              .evaluate(parameters) op returnOperator._operators_to_compose[1] \
              .get_val();                                                      \
        };                                                                     \
    returnOperator.generator = scalar_callback_function(newGenerator);         \
    return returnOperator;                                                     \
  }

#define ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(op)                              \
  ScalarOperator operator op(ScalarOperator self,                              \
                             std::complex<double> other) {                     \
    /* Create an operator for the complex double value. */                     \
    auto otherOperator = ScalarOperator(other);                                \
    /* Create an operator that we will store the result in and return to the   \
     * user. */                                                                \
    ScalarOperator returnOperator;                                             \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    returnOperator._operators_to_compose.push_back(self);                      \
    returnOperator._operators_to_compose.push_back(otherOperator);             \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          /* FIXME: I have to use this hacky `.get_val()` on the newly created \
           * operator for the given complex double -- because calling the      \
           * evaluate function returns 0.0 . I have no clue why??? */          \
          return returnOperator._operators_to_compose[1]                       \
              .get_val() op returnOperator._operators_to_compose[0]            \
              .evaluate(parameters);                                           \
        };                                                                     \
    returnOperator.generator = scalar_callback_function(newGenerator);         \
    return returnOperator;                                                     \
  }

#define ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(op)                           \
  void operator op(ScalarOperator &self, std::complex<double> other) {         \
    /* Create an operator for the complex double value. */                     \
    auto otherOperator = ScalarOperator(other);                                \
    /* Need to move the existing generating function to a new operator so that \
     * we can modify the generator in `self` in-place. */                      \
    ScalarOperator copy(self);                                                 \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    self._operators_to_compose.push_back(copy);                                \
    self._operators_to_compose.push_back(otherOperator);                       \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          /* FIXME: I have to use this hacky `.get_val()` on the newly created \
           * operator for the given complex double -- because calling the      \
           * evaluate function returns 0.0 . I have no clue why??? */          \
          return self._operators_to_compose[0]                                 \
              .evaluate(parameters) op self._operators_to_compose[1]           \
              .get_val();                                                      \
        };                                                                     \
    self.generator = scalar_callback_function(newGenerator);                   \
  }

ARITHMETIC_OPERATIONS_DOUBLES(+);
ARITHMETIC_OPERATIONS_DOUBLES(-);
ARITHMETIC_OPERATIONS_DOUBLES(*);
ARITHMETIC_OPERATIONS_DOUBLES(/);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(-);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(/);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(/=);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS(op)                                   \
  ScalarOperator ScalarOperator::operator op(ScalarOperator other) {           \
    /* Create an operator that we will store the result in and return to the   \
     * user. */                                                                \
    ScalarOperator returnOperator;                                             \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    returnOperator._operators_to_compose.push_back(*this);                     \
    returnOperator._operators_to_compose.push_back(other);                     \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return returnOperator._operators_to_compose[0]                       \
              .evaluate(parameters) op returnOperator._operators_to_compose[1] \
              .evaluate(parameters);                                           \
        };                                                                     \
    returnOperator.generator = scalar_callback_function(newGenerator);         \
    return returnOperator;                                                     \
  }

/// FIXME: Broken implementation
#define ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(op)                        \
  void operator op(ScalarOperator &self, ScalarOperator other) {               \
    /* Need to move the existing generating function to a new operator so      \
     * that we can modify the generator in `self` in-place. */                 \
    ScalarOperator selfCopy(self);                                             \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    self._operators_to_compose.push_back(selfCopy);                            \
    self._operators_to_compose.push_back(other);                               \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return self._operators_to_compose[0]                                 \
              .evaluate(parameters) op self._operators_to_compose[1]           \
              .evaluate(parameters);                                           \
        };                                                                     \
    self.generator = scalar_callback_function(newGenerator);                   \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS(-);
ARITHMETIC_OPERATIONS_SCALAR_OPS(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS(/);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(/=);

} // namespace cudaq