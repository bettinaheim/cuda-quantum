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

#include <numeric>

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

cudaq::complex_matrix kroneckerHelper(std::vector<cudaq::complex_matrix> &matrices) {
  // essentially we pass in the list of elementary operators to
  // this function -- with lowest degree being leftmost -- then it computes the
  // kronecker product of all of them in reverse order.
  // E.g, if degrees = {0, 1, 2}, it computes as `matrix(2) kron matrix(1) kron matrix(0)`.
  auto kronecker = [](cudaq::complex_matrix self, cudaq::complex_matrix other) { 
    return cudaq::kronecker(self,other);
  };
  return std::reduce(begin(matrices), end(matrices), cudaq::complex_matrix::identity(1,1), kronecker);
}

// TEST(ExpressionTester, checkKronecker) {
//   {
//     std::cout << "\n\n\n\n hello \n\n\n\n";
//     auto id0 = cudaq::complex_matrix::identity(2,2);
//     auto id1 = cudaq::complex_matrix::identity(1,1);
//     auto id2 = cudaq::complex_matrix::identity(1,1);
//     auto id3 = cudaq::complex_matrix::identity(1,1);

//     std::vector<cudaq::complex_matrix> matrices = {id0, id1, id2, id3};

//     auto kronecker = [](cudaq::complex_matrix self, cudaq::complex_matrix other) { 
//       // return self.kronecker(other);
//       return cudaq::kronecker(self,other);
//     };

//     // auto start = cudaq::complex_matrix::identity(1,1);
//     // auto new_ = kronecker(start, id0);
//     // auto new_1 = kronecker(new_, id0);

//     // std::cout << "\n";
//     // new_1.dump();
//     // std::cout << "\n";

//     std::vector<cudaq::complex_matrix> kronMats = {cudaq::complex_matrix::identity(1,1)};
//     auto start = cudaq::complex_matrix::identity(1,1);
//     for (auto &matrix : matrices) {
//       start = kronecker(kronMats[idx], matrix);
//     }
//     start.dump();


//     // // auto got_matrix = kroneckerHelper(matrices);
//     // // auto want_matrix = cudaq::complex_matrix::identity(16,16);

//     // // std::cout << "\n";
//     // // got_matrix.dump();
//     // // std::cout << "\n";

//     // // ASSERT_TRUE(want_matrix == got_matrix);
//   }
// }

/// TODO: Not yet testing the output matrices coming from this arithmetic.

TEST(ExpressionTester, checkProductOperatorElementaryOnly) {
  std::vector<int> levels = {2, 3, 4};

  {
    // Same degrees of freedom.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(0);

        cudaq::product_operator got = op0 * op1;

        // auto got_matrix = got.to_matrix({{0, level_count}}, {});

        // auto matrix0 = _annihilate_matrix(level_count);
        // auto matrix1 = _create_matrix(level_count);
        // auto want_matrix = matrix0 * matrix1;

        // ASSERT_TRUE(want_matrix == got_matrix);

        std::vector<int> want_degrees = {0};
        ASSERT_TRUE(got.degrees() == want_degrees);
      }
    }

    // Different degrees of freedom.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(1);

        cudaq::product_operator got = op0 * op1;
        // auto got_matrix =
        //     got.to_matrix({{0, level_count}, {1, level_count}}, {});

        cudaq::product_operator got_reverse = op1 * op0;
        // auto got_matrix_reverse =
        //     got_reverse.to_matrix({{0, level_count}, {1, level_count}}, {});

        // auto identity = _id_matrix(level_count);
        // auto matrix0 = _annihilate_matrix(level_count);
        // auto matrix1 = _create_matrix(level_count);

        // auto fullHilbert0 = identity.kronecker(matrix0);
        // auto fullHilbert1 = matrix1.kronecker(identity);
        // auto want_matrix = fullHilbert0 * fullHilbert1;
        // auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

        // ASSERT_TRUE(want_matrix == got_matrix);
        // ASSERT_TRUE(want_matrix_reverse == got_matrix_reverse);

        std::vector<int> want_degrees = {0, 1};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);
      }
    }

    // Different degrees of freedom, non-consecutive.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(2);

        cudaq::product_operator got = op0 * op1;
        // auto got_matrix = got.to_matrix({{0,level_count},{2,level_count}},
        // {});

        cudaq::product_operator got_reverse = op1 * op0;

        std::vector<int> want_degrees = {0, 2};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);
      }
    }

    // Different degrees of freedom, non-consecutive but all dimensions
    // provided.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(2);

        cudaq::product_operator got = op0 * op1;
        // auto got_matrix =
        // got.to_matrix({{0,level_count},{1,level_count},{2,level_count}}, {});

        cudaq::product_operator got_reverse = op1 * op0;

        std::vector<int> want_degrees = {0, 2};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);
      }
    }
  }
}

TEST(ExpressionTester, checkProductOperatorMixed) {

  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  auto local_variable = true;
  auto function = [&](std::map<std::string, std::complex<double>> parameters)
  {
    if (!local_variable)
      throw std::runtime_error("Local variable not detected.");
    return parameters["value"];
  };

  // Scalar Ops against Elementary Ops
  {
    // Annihilation against constant.
    {
      auto id_op = cudaq::elementary_operator::annihilate(0);
      auto scalar_op = cudaq::scalar_operator(value_0);

      auto got = scalar_op * id_op;
      auto got_reverse = scalar_op * id_op;

      std::vector<int> want_degrees = {0};
      ASSERT_TRUE(got.degrees() == want_degrees);
      ASSERT_TRUE(got_reverse.degrees() == want_degrees);
    }

    // Annihilation against constant from lambda.
    {
      auto id_op = cudaq::elementary_operator::annihilate(1);
      auto scalar_op = cudaq::scalar_operator(function);

      auto got = scalar_op * id_op;
      auto got_reverse = scalar_op * id_op;

      std::vector<int> want_degrees = {1};
      ASSERT_TRUE(got.degrees() == want_degrees);
      ASSERT_TRUE(got_reverse.degrees() == want_degrees);
    }
  }
}