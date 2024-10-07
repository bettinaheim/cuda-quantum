/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>
#include "cudaq/matrix.h"
#include "cudaq/operators.h"

cudaq::complex_matrix zero_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  mat.set_zero();
  return mat;
}

cudaq::complex_matrix id_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (int i = 0; i < size; i++)
    mat(i, i) = 1.0 + 0.0j;
  return mat;
}

cudaq::complex_matrix annihilate_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat(i, i + 1) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::complex_matrix create_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat(i + 1, i) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::complex_matrix position_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat(i + 1, i) = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat(i, i + 1) = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::complex_matrix momentum_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat(i + 1, i) = (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat(i, i + 1) =
        -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::complex_matrix number_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (int i = 0; i < size; i++)
    mat(i, i) = static_cast<double>(i) + 0.0j;
  return mat;
}

cudaq::complex_matrix parity_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (int i = 0; i < size; i++)
    mat(i, i) = std::pow(-1., static_cast<double>(i)) + 0.0j;
  return mat;
}

cudaq::complex_matrix displace_matrix(int size,
                                      std::complex<double> amplitude) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat(i + 1, i) =
        amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat(i, i + 1) = -1. * std::conj(amplitude) * (0.5 * 'j') *
                        std::sqrt(static_cast<double>(i + 1)) +
                    0.0 * 'j';
  }
  return mat.exp();
}

TEST(ExpressionTester, checkPreBuiltElementaryOps) {
  std::vector<int> sizes = {2, 3, 4, 5};

  // Keeping this fixed throughout.
  int degree_index = 0;

  // Identity operator.
  {
    for (int size : sizes) {
      // cudaq::operators::identity(int degree)
      auto id = cudaq::elementary_operator::identity(degree_index);
      auto got_id = id.to_matrix({{degree_index, size}}, {});
      auto want_id = id_matrix(size);
      ASSERT_TRUE(want_id == got_id);
    }
  }

  // Zero operator.
  {
    for (int size : sizes) {
      auto zero = cudaq::elementary_operator::zero(degree_index);
      auto got_zero = zero.to_matrix({{degree_index, size}}, {});
      auto want_zero = zero_matrix(size);
      ASSERT_TRUE(want_zero == got_zero);
    }
  }

  // Annihilation operator.
  {
    for (int size : sizes) {
      auto annihilate = cudaq::elementary_operator::annihilate(degree_index);
      auto got_annihilate = annihilate.to_matrix({{degree_index, size}}, {});
      auto want_annihilate = annihilate_matrix(size);
      ASSERT_TRUE(want_annihilate == got_annihilate);
    }
  }

  // Creation operator.
  {
    for (int size : sizes) {
      auto create = cudaq::elementary_operator::create(degree_index);
      auto got_create = create.to_matrix({{degree_index, size}}, {});
      auto want_create = create_matrix(size);
      ASSERT_TRUE(want_create == got_create);
    }
  }

  // Position operator.
  {
    for (int size : sizes) {
      auto position = cudaq::elementary_operator::position(degree_index);
      auto got_position = position.to_matrix({{degree_index, size}}, {});
      auto want_position = position_matrix(size);
      ASSERT_TRUE(want_position == got_position);
    }
  }

  // Momentum operator.
  {
    for (int size : sizes) {
      auto momentum = cudaq::elementary_operator::momentum(degree_index);
      auto got_momentum = momentum.to_matrix({{degree_index, size}}, {});
      auto want_momentum = momentum_matrix(size);
      ASSERT_TRUE(want_momentum == got_momentum);
    }
  }

  // Number operator.
  {
    for (int size : sizes) {
      auto number = cudaq::elementary_operator::number(degree_index);
      auto got_number = number.to_matrix({{degree_index, size}}, {});
      auto want_number = number_matrix(size);
      ASSERT_TRUE(want_number == got_number);
    }
  }

  // Parity operator.
  {
    for (int size : sizes) {
      auto parity = cudaq::elementary_operator::parity(degree_index);
      auto got_parity = parity.to_matrix({{degree_index, size}}, {});
      auto want_parity = parity_matrix(size);
      ASSERT_TRUE(want_parity == got_parity);
    }
  }

  // // Displacement operator.
  // {
  //   for (int size : sizes) {
  //     auto amplitude = 1.0 + 1.0j;
  //     auto displace = cudaq::elementary_operator::displace(degree_index,
  //     amplitude); auto got_displace = displace.to_matrix({{degree_index,
  //     size}}, {}); auto want_displace = displace_matrix(size, amplitude);
  //     ASSERT_TRUE(want_displace == got_displace);
  //   }
  // }

  // TODO: Squeeze operator.
}

// TEST(ExpressionTester, checkCustomElementaryOps) {
//   // pass
// }