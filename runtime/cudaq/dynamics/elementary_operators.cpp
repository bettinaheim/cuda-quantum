/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "helpers.cpp"

#include <iostream>
#include <set>

namespace cudaq {

/// Elementary Operator constructor.
elementary_operator::elementary_operator(std::string operator_id,
                                         std::vector<int> degrees)
    : id(operator_id),
      degrees(_OperatorHelpers::canonicalize_degrees(degrees)) {}
elementary_operator::elementary_operator(const elementary_operator &other)
    : m_ops(other.m_ops), expected_dimensions(other.expected_dimensions),
      degrees(other.degrees), id(other.id) {}
elementary_operator::elementary_operator(elementary_operator &other)
    : m_ops(other.m_ops), expected_dimensions(other.expected_dimensions),
      degrees(other.degrees), id(other.id) {}

elementary_operator elementary_operator::identity(int degree) {
  std::string op_id = "identity";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      int degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);

      // Build up the identity matrix.
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = 1.0 + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::zero(int degree) {
  std::string op_id = "zero";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::annihilate(int degree) {
  std::string op_id = "annihilate";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::create(int degree) {
  std::string op_id = "create";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::position(int degree) {
  std::string op_id = "position";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      // position = 0.5 * (create + annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] =
            0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat[{i, i + 1}] =
            0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::momentum(int degree) {
  std::string op_id = "momentum";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      // momentum = 0.5j * (create - annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] =
            (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat[{i, i + 1}] =
            -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::number(int degree) {
  std::string op_id = "number";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = static_cast<double>(i) + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::parity(int degree) {
  std::string op_id = "parity";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, op](std::map<int, int> dimensions,
                        std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::displace(int degree) {
  std::string op_id = "displace";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&,
                 op](std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto displacement_amplitude = parameters["displacement"];
      auto create = matrix_2(dimension, dimension);
      auto annihilate = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        annihilate[{i, i + 1}] =
            std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      auto term1 = displacement_amplitude * create;
      auto term2 = std::conj(displacement_amplitude) * annihilate;
      return (term1 - term2).exponential();
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::squeeze(int degree) {
  std::string op_id = "squeeze";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&,
                 op](std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      auto degree = op.degrees[0];
      std::size_t dimension = dimensions[degree];
      auto squeezing = parameters["squeezing"];
      auto create = matrix_2(dimension, dimension);
      auto annihilate = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        annihilate[{i, i + 1}] =
            std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      auto term1 = std::conj(squeezing) * annihilate.power(2);
      auto term2 = squeezing * create.power(2);
      auto difference = 0.5 * (term1 - term2);
      return difference.exponential();
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

matrix_2 elementary_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {
  /// TODO: Do a dimension check on the passed in degrees for safety.
  return m_ops[id].generator(dimensions, parameters);
}

/// Elementary Operator Arithmetic.

operator_sum elementary_operator::operator+(scalar_operator &other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum({product_operator({}, {*this}), product_operator({other}, {})});
}

/// FIXME:
operator_sum elementary_operator::operator-(scalar_operator &other) {
  std::cout << "\n in elementary - scalar \n";
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  auto result = operator_sum({product_operator({}, {*this}), product_operator({-1.0 * other}, {})});
  std::cout << "\n first term prod = \n" << product_operator({}, {*this}).to_matrix({{0,3}}).dump() << "\n";
  std::cout << "\n second term prod = \n" << product_operator({-1.0 * other}, {}).to_matrix({{0,3}}).dump() << "\n";
  std::cout << "\n result matrix = \n" << result.to_matrix({{0,3}}).dump() << "\n";
  return result;
  // return operator_sum({product_operator({}, {*this}), product_operator({-1.0 * other}, {})});
}

product_operator elementary_operator::operator*(scalar_operator &other) {
  return product_operator({other}, {*this});
}

operator_sum elementary_operator::operator+(std::complex<double> other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  auto other_scalar = scalar_operator(other);
  return operator_sum({product_operator({}, {*this}), product_operator({other_scalar}, {})});
}

operator_sum elementary_operator::operator-(std::complex<double> other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  auto other_scalar = scalar_operator((-1. * other));
  std::vector<elementary_operator> _this = {*this};
  std::vector<scalar_operator> _other = {other_scalar};
  return operator_sum({product_operator({}, _this), product_operator(_other, {})});
}

product_operator elementary_operator::operator*(std::complex<double> other) {
  auto other_scalar = scalar_operator(other);
  std::vector<elementary_operator> _this = {*this};
  std::vector<scalar_operator> _other = {other_scalar};
  return product_operator(_other, _this);
}

operator_sum elementary_operator::operator+(double other) {
  std::complex<double> value(other, 0.0);
  return *this + value;
}

operator_sum elementary_operator::operator-(double other) {
  std::complex<double> value(other, 0.0);
  return *this - value;
}

product_operator elementary_operator::operator*(double other) {
  std::complex<double> value(other, 0.0);
  return *this * value;
}

operator_sum operator+(std::complex<double> other, elementary_operator &self) {
  auto other_scalar = scalar_operator(other);
  std::vector<elementary_operator> _self = {self};
  std::vector<scalar_operator> _other = {other_scalar};
  return operator_sum({product_operator(_other, {}), product_operator({}, _self)});
}

operator_sum operator-(std::complex<double> other, elementary_operator &self) {
  auto other_scalar = scalar_operator(other);
  std::vector<scalar_operator> _other = {other_scalar};
  return operator_sum({product_operator(_other, {}), (-1. * self)});
}

product_operator operator*(std::complex<double> other,
                           elementary_operator &self) {
  auto other_scalar = scalar_operator(other);
  std::vector<elementary_operator> _self = {self};
  std::vector<scalar_operator> _other = {other_scalar};
  return product_operator(_other, _self);
}

operator_sum operator+(double other, elementary_operator &self) {
  auto other_scalar = scalar_operator(other);
  std::vector<elementary_operator> _self = {self};
  std::vector<scalar_operator> _other = {other_scalar};
  return operator_sum({product_operator({_other, {}}), product_operator({}, _self)});
}

operator_sum operator-(double other, elementary_operator &self) {
  auto other_scalar = scalar_operator(other);
  std::vector<scalar_operator> _other = {other_scalar};
  return operator_sum({product_operator(_other, {}), (-1. * self)});
}

product_operator operator*(double other, elementary_operator &self) {
  auto other_scalar = scalar_operator(other);
  std::vector<elementary_operator> _self = {self};
  std::vector<scalar_operator> _other = {other_scalar};
  return product_operator(_other, _self);
}

product_operator elementary_operator::operator*(elementary_operator other) {
  std::vector<elementary_operator> _self = {*this, other};
  return product_operator({}, _self);
}

operator_sum elementary_operator::operator+(elementary_operator other) {
  std::vector<elementary_operator> _this = {*this};
  std::vector<elementary_operator> _other = {other};
  return operator_sum({product_operator({}, _this), product_operator({}, _other)});
}

operator_sum elementary_operator::operator-(elementary_operator other) {
  std::vector<elementary_operator> _this = {*this};
  return operator_sum({product_operator({}, _this), (-1. * other)});
}

operator_sum elementary_operator::operator+(operator_sum &other) {
  std::vector<product_operator> _prods = {product_operator({}, {*this})};
  auto selfOpSum = operator_sum(_prods);
  return selfOpSum + other;
}

operator_sum elementary_operator::operator-(operator_sum &other) {
  std::vector<product_operator> _prods = {product_operator({}, {*this})};
  auto selfOpSum = operator_sum(_prods);
  return selfOpSum - other;
}

operator_sum elementary_operator::operator*(operator_sum &other) {
  std::vector<product_operator> _prods = {product_operator({}, {*this})};
  auto selfOpSum = operator_sum(_prods);
  return selfOpSum * other;
}

operator_sum elementary_operator::operator+(product_operator &other) {
  return operator_sum({product_operator({}, {*this}), other});
}

operator_sum elementary_operator::operator-(product_operator &other) {
  product_operator negative_other(other);
  negative_other *= -1.0;
  return *this + negative_other;
}

product_operator elementary_operator::operator*(product_operator &other) {
  std::vector<elementary_operator> other_elementary_ops = other.m_elementary_ops;
  /// Insert this elementary operator to the front of the terms list.
  other_elementary_ops.insert(other_elementary_ops.begin(), *this);
  return product_operator(m_scalar_ops, other_elementary_ops);
}

} // namespace cudaq