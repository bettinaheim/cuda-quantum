/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/matrix.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>

cudaq::complex_matrix _id_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (int i = 0; i < size; i++)
    mat(i, i) = 1.0 + 0.0j;
  return mat;
}

cudaq::complex_matrix _annihilate_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat(i, i + 1) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::complex_matrix _create_matrix(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat(i + 1, i) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

TEST(ExpressionTester, checkProductOperatorSimple) {
  std::vector<int> levels = {2, 3, 4};

  // std::set<int> uniqueDegrees;
  // std::copy(this->degrees.begin(), this->degrees.end(), std::inserter(uniqueDegrees, uniqueDegrees.begin()));
  // std::copy(other.degrees.begin(), other.degrees.end(), std::inserter(uniqueDegrees, uniqueDegrees.begin()));

  // Arithmetic only between elementary operators with
  // same number of levels.
  {
    // Same degrees of freedom.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(0);

        cudaq::product_operator got = op0 * op1;
        auto got_matrix = got.to_matrix({{0, level_count}}, {});

        auto matrix0 = _annihilate_matrix(level_count);
        auto matrix1 = _create_matrix(level_count);
        auto want_matrix = matrix0 * matrix1;

        // ASSERT_TRUE(want_matrix == got_matrix);
      }
    }

    // // Different degrees of freedom.
    // {
    //   for (auto level_count : levels) {
    //     auto op0 = cudaq::elementary_operator::annihilate(0);
    //     auto op1 = cudaq::elementary_operator::create(1);

    //     cudaq::product_operator got = op0 * op1;
    //     auto got_matrix =
    //         got.to_matrix({{0, level_count}, {1, level_count}}, {});

    //     cudaq::product_operator got_reverse = op1 * op0;
    //     auto got_matrix_reverse =
    //         got_reverse.to_matrix({{0, level_count}, {1, level_count}}, {});

    //     auto identity = _id_matrix(level_count);
    //     auto matrix0 = _annihilate_matrix(level_count);
    //     auto matrix1 = _create_matrix(level_count);

    //     auto fullHilbert0 = identity.kronecker(matrix0);
    //     auto fullHilbert1 = matrix1.kronecker(identity);
    //     auto want_matrix = fullHilbert0 * fullHilbert1;
    //     auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

    //     // ASSERT_TRUE(want_matrix == got_matrix);
    //     // ASSERT_TRUE(want_matrix_reverse == got_matrix_reverse);
    //   }
    // }

    // // Different degrees of freedom, non-consecutive.
    // {
    //   for (auto level_count : levels) {
    //     auto op0 = cudaq::elementary_operator::annihilate(0);
    //     auto op1 = cudaq::elementary_operator::create(2);

    //     // cudaq::product_operator got = op0 * op1;
    //     // auto got_matrix = got.to_matrix({{0,level_count},{2,level_count}},
    //     // {});
    //   }
    // }

    // // Different degrees of freedom, non-consecutive but all dimensions
    // // provided.
    // {
    //   for (auto level_count : levels) {
    //     auto op0 = cudaq::elementary_operator::annihilate(0);
    //     auto op1 = cudaq::elementary_operator::create(2);

    //     // cudaq::product_operator got = op0 * op1;
    //     // auto got_matrix =
    //     // got.to_matrix({{0,level_count},{1,level_count},{2,level_count}}, {});
    //   }
    // }
  }
}

// TEST(ExpressionTester, checkProductOperatorSimple) {

//   std::complex<double> value_0 = 0.1 + 0.1;
//   std::complex<double> value_1 = 0.1 + 1.0;
//   std::complex<double> value_2 = 2.0 + 0.1;
//   std::complex<double> value_3 = 2.0 + 1.0;

//   auto local_variable = true;
//   auto function = [&](std::map<std::string, std::complex<double>> parameters)
//   {
//     if (!local_variable)
//       throw std::runtime_error("Local variable not detected.");
//     return parameters["value"];
//   };

//   // Scalar Ops against Elementary Ops
//   {
//     // Identity against constant.
//     {
//       auto id_op = cudaq::elementary_operator::identity(0);
//       auto scalar_op = cudaq::scalar_operator(value_0);

//       // auto multiplication = scalar_op * id_op;
//       // auto addition = scalar_op + id_op;
//       // auto subtraction = scalar_op - id_op;
//     }

//     // Identity against constant from lambda.
//     {
//       auto id_op = cudaq::elementary_operator::identity(0);
//       auto scalar_op = cudaq::scalar_operator(function);

//       // auto multiplication = scalar_op * id_op;
//       // auto addition = scalar_op + id_op;
//       // auto subtraction = scalar_op - id_op;
//     }
//   }
// }