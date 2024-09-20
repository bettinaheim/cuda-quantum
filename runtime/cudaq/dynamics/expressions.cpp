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

// implement everything without `_evaluate` first

// OperatorSum(std::vector<ProductOperator> &terms) {
//   m_terms = terms;
// }

// TODO:
// (1) Elementary Operators
// (2) Scalar Operators

ElementaryOperator::ElementaryOperator(std::string operator_id,
                                       std::vector<int> degrees)
    : id(operator_id), degrees(degrees) {}

ElementaryOperator ElementaryOperator::identity(int degree) {
  std::string op_id = "identity";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::vector<int> none, std::vector<Parameter> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      auto mat = complex_matrix(degree, degree);
      // Build up the identity matrix.
      for (std::size_t i = 0; i < degree; i++) {
        mat(i, i) = 1.0 + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, degrees, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::zero(int degree) {
  std::string op_id = "zero";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::vector<int> none, std::vector<Parameter> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      auto mat = complex_matrix(degree, degree);
      mat.set_zero();
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, degrees, func);
  }
  return op;
}

complex_matrix
ElementaryOperator::to_matrix(std::vector<int> degrees,
                              std::vector<Parameter> parameters) {
  ReturnType result = m_ops[id].m_generator(degrees, parameters);

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
    // : generator(other.generator), m_constant_value(other.m_constant_value) {}
{
  // generator = new scalar_callback_function;
  generator = other.generator;
  m_constant_value = other.m_constant_value;
  std::copy(other.m_parameters.begin(), other.m_parameters.end(), std::back_inserter(m_parameters));
}
ScalarOperator::ScalarOperator(ScalarOperator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}
ScalarOperator::ScalarOperator(ScalarOperator &&other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}

/// @FIXME: The below function signature can be updated once
/// we support generalized function arguments.
/// @brief Constructor that just takes and returns a complex double value.
ScalarOperator::ScalarOperator(std::complex<double> value) {
  m_constant_value = value;
  std::cout << "\nm_constant_value = " << m_constant_value << "\n";
  auto func = [&](std::vector<std::complex<double>> _none) {
    std::cout << "\nm_constant_value in fn call = " << m_constant_value << "\n";
    return m_constant_value;
  };
  generator = scalar_callback_function(func);
}

std::complex<double>
ScalarOperator::evaluate(std::vector<std::complex<double>> parameters) {
  return generator(parameters);
}

// Arithmetic Operations.
ScalarOperator operator+(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    /// NOTE: This fails the unit test if I reverse the order of the evaluate
    /// calls???
    return returnOperator._operators_to_compose[0].evaluate(selfParams) +
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator+(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) +
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator-(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) -
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator-(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[1].evaluate({}) -
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator*(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) *
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator*(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    // std::cout << "\n\nfirst value = " <<
    // returnOperator._operators_to_compose[1].evaluate({}) << "\n"; std::cout
    // << "second value = " <<
    // returnOperator._operators_to_compose[0].evaluate(selfParams) << "\n";
    // std::cout << "product = " <<
    // returnOperator._operators_to_compose[1].evaluate({}) *
    // returnOperator._operators_to_compose[0].evaluate(selfParams) << "\n\n\n";
    std::cout << "\naccessing via evaluate = "
              << returnOperator._operators_to_compose[1].evaluate({}) << "\n";
    std::cout << "\naccessing via generator = "
              << returnOperator._operators_to_compose[1].generator({}) << "\n";
    return returnOperator._operators_to_compose[1].evaluate({}) *
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  // newGenerator({});

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator/(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) /
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator/(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[1].evaluate({}) /
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

void operator+=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) +
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

void operator-=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) -
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

void operator*=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) *
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

void operator/=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) /
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

} // namespace cudaq