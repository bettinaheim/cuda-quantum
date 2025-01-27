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
#include <complex>
#include <set>

namespace cudaq {

std::map<std::string, Definition> elementary_operator::m_ops = {};

product_operator<elementary_operator> elementary_operator::identity(int degree) {
  std::string op_id = "identity";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);

      // Build up the identity matrix.
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = 1.0 + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::zero(int degree) {
  std::string op_id = "zero";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::annihilate(int degree) {
  std::string op_id = "annihilate";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::create(int degree) {
  std::string op_id = "create";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::position(int degree) {
  std::string op_id = "position";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
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
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::momentum(int degree) {
  std::string op_id = "momentum";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
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
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::number(int degree) {
  std::string op_id = "number";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = static_cast<double>(i) + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::parity(int degree) {
  std::string op_id = "parity";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator>
elementary_operator::displace(int degree, std::complex<double> amplitude) {
  std::string op_id = "displace";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  // if (op.m_ops.find(op_id) == op.m_ops.end()) {
  //   auto func = [&, degree](std::map<int, int> dimensions,
  //                   std::map<std::string, std::complex<double>> _none) {
  //     std::size_t dimension = dimensions[degree];
  //     auto temp_mat = matrix_2(dimension, dimension);
  //     // // displace = exp[ (amplitude * create) - (conj(amplitude) *
  //     annihilate) ]
  //     // for (std::size_t i = 0; i + 1 < dimension; i++) {
  //     //   temp_mat[{i + 1, i}] =
  //     //       amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  //     //   temp_mat[{i, i + 1}] =
  //     //       -1. * std::conj(amplitude) * std::sqrt(static_cast<double>(i +
  //     1)) +
  //     //       0.0 * 'j';
  //     // }
  //     // Not ideal that our method of computing the matrix exponential
  //     // requires copies here. Maybe we can just use eigen directly here
  //     // to limit to one copy, but we can address that later.
  //     auto mat = temp_mat.exp();
  //     return mat;
  //   };
  //   op.define(op_id, op.expected_dimensions, func);
  // }
  throw std::runtime_error("currently have a bug in implementation.");
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator>
elementary_operator::squeeze(int degree, std::complex<double> amplitude) {
  throw std::runtime_error("Not yet implemented.");
}

// evaluations

matrix_2 elementary_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {
  return m_ops[id].generator(dimensions, parameters);
}

} // namespace cudaq