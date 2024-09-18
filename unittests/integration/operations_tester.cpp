/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/expressions.h"

cudaq::complex_matrix zero(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  mat.set_zero();
  return mat;
}

cudaq::complex_matrix id(int size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (int i = 0; i < size; i++)
    mat(i, i) = 1.0 + 0.0j;
  return mat;
}

TEST(ExpressionTester, checkPreBuiltElementaryOps) {
  // Check 2x2 variations.
  {
    auto id_2 = cudaq::ElementaryOperator::identity(2);
    auto zero_2 = cudaq::ElementaryOperator::zero(2);

    auto want_id_2 = id_2.to_matrix({}, {});
    auto want_zero_2 = zero_2.to_matrix({}, {});
    auto got_id_2 = id(2);
    auto got_zero_2 = zero(2);

    ASSERT_TRUE(want_id_2 == got_id_2);
    ASSERT_TRUE(want_zero_2 == got_zero_2);
  }

  // Check 3x3 variations.
  {
    auto id_3 = cudaq::ElementaryOperator::identity(3);
    auto zero_3 = cudaq::ElementaryOperator::zero(3);

    auto want_id_3 = id_3.to_matrix({}, {});
    auto want_zero_3 = zero_3.to_matrix({}, {});
    auto got_id_3 = id(3);
    auto got_zero_3 = zero(3);

    ASSERT_TRUE(want_id_3 == got_id_3);
    ASSERT_TRUE(want_zero_3 == got_zero_3);
  }

  // Check 4x4 variations.
  {
    auto id_4 = cudaq::ElementaryOperator::identity(4);
    auto zero_4 = cudaq::ElementaryOperator::zero(4);

    auto want_id_4 = id_4.to_matrix({}, {});
    auto want_zero_4 = zero_4.to_matrix({}, {});
    auto got_id_4 = id(4);
    auto got_zero_4 = zero(4);

    ASSERT_TRUE(want_id_4 == got_id_4);
    ASSERT_TRUE(want_zero_4 == got_zero_4);
  }
}

TEST(ExpressionTester, checkCustomElementaryOps) {
  // pass
}

TEST(ExpressionTester, checkScalarOps) {

  /// TODO: Test the following overload once we have general
  /// parameters support:
  /*
    ScalarOperator(Callable &&create,
                 std::map<std::string, std::complex<double>> parameters) {
  */

  std::complex<double> want_value_0 = 0.0 + 0.0;
  std::complex<double> want_value_1 = 0.0 + 1.0;
  std::complex<double> want_value_2 = 2.0 + 0.0;
  std::complex<double> want_value_3 = 2.0 + 1.0;

  // From concrete values.
  {
    auto got_value_0 = cudaq::ScalarOperator(want_value_0).evaluate({});
    auto got_value_1 = cudaq::ScalarOperator(want_value_1).evaluate({});
    auto got_value_2 = cudaq::ScalarOperator(want_value_2).evaluate({});
    auto got_value_3 = cudaq::ScalarOperator(want_value_3).evaluate({});

    EXPECT_NEAR(std::abs(want_value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(want_value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(want_value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(want_value_3), std::abs(got_value_3), 1e-5);
  }

  // From a lambda function.
  {
    auto function = [](std::vector<std::complex<double>> vec) {
      return vec[0];
    };
    auto got_value_0 = cudaq::ScalarOperator(function).evaluate({want_value_0});
    auto got_value_1 = cudaq::ScalarOperator(function).evaluate({want_value_1});
    auto got_value_2 = cudaq::ScalarOperator(function).evaluate({want_value_2});
    auto got_value_3 = cudaq::ScalarOperator(function).evaluate({want_value_3});

    EXPECT_NEAR(std::abs(want_value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(want_value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(want_value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(want_value_3), std::abs(got_value_3), 1e-5);
  }

  /// FIXME: All of the below are segfaulting directly on the
  /// arithmetic call -- even if the actual arithmetic operator
  /// overload has an empty function body containing a `throw`.

  // Arithmetic overloads against complex doubles.
  {
    // + : Constant scalar operator.
    {
      auto scalar_op = cudaq::ScalarOperator(want_value_0);
      auto new_scalar_op = want_value_0 + scalar_op;

      auto scalar_op_1 = cudaq::ScalarOperator(want_value_0);
      auto reverse_order_op = scalar_op_1 + want_value_0;

      auto got_value = new_scalar_op.evaluate({});
      auto got_value_1 = reverse_order_op.evaluate({});

      EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 + want_value_0),
                  1e-5);
      EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_0 + want_value_0),
                  1e-5);
    }

    // + : Scalar operator from lambda.
    {
      auto function = [](std::vector<std::complex<double>> vec) {
        return vec[0];
      };
      auto scalar_op = cudaq::ScalarOperator(function);
      auto new_scalar_op = want_value_0 + scalar_op;

      auto scalar_op_1 = cudaq::ScalarOperator(function);
      auto reverse_order_op = scalar_op_1 + want_value_0;

      auto got_value = new_scalar_op.evaluate({want_value_1});
      auto got_value_1 = reverse_order_op.evaluate({want_value_1});

      EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 + want_value_1),
                  1e-5);
      EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_1 + want_value_0),
                  1e-5);
    }

    // - : Constant scalar operator.
    {
      auto scalar_op = cudaq::ScalarOperator(want_value_1);
      auto new_scalar_op = want_value_0 - scalar_op;

      auto scalar_op_1 = cudaq::ScalarOperator(want_value_1);
      auto reverse_order_op = scalar_op_1 - want_value_0;

      auto got_value = new_scalar_op.evaluate({});
      auto got_value_1 = reverse_order_op.evaluate({});

      EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 - want_value_1),
                  1e-5);
      EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_1 - want_value_0),
                  1e-5);
    }

    // - : Scalar operator from lambda.
    {
      auto function = [](std::vector<std::complex<double>> vec) {
        return vec[0];
      };
      auto scalar_op = cudaq::ScalarOperator(function);
      auto new_scalar_op = want_value_2 - scalar_op;

      auto scalar_op_1 = cudaq::ScalarOperator(function);
      auto reverse_order_op = scalar_op_1 - want_value_2;

      auto got_value = new_scalar_op.evaluate({want_value_1});
      auto got_value_1 = reverse_order_op.evaluate({want_value_1});

      EXPECT_NEAR(std::abs(got_value), std::abs(want_value_2 - want_value_1),
                  1e-5);
      EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_1 - want_value_2),
                  1e-5);
    }

    // * : Constant scalar operator.
    {
      auto scalar_op = cudaq::ScalarOperator(want_value_2);
      auto new_scalar_op = want_value_3 * scalar_op;

      // auto scalar_op_1 = cudaq::ScalarOperator(want_value_2);
      // auto reverse_order_op = scalar_op_1 * want_value_3;

      auto got_value = new_scalar_op.evaluate({});
      std::cout << "got value"
      // auto got_value_1 = reverse_order_op.evaluate({});

      EXPECT_NEAR(std::abs(got_value), std::abs(want_value_3 * want_value_2),
                  1e-5);
      // EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_2 * want_value_3),
      //             1e-5);
    }

    // // * : Scalar operator from lambda.
    // {
    //   auto function = [](std::vector<std::complex<double>> vec) {
    //     return vec[0];
    //   };
    //   auto scalar_op = cudaq::ScalarOperator(function);
    //   auto new_scalar_op = want_value_3 * scalar_op;

    //   auto scalar_op_1 = cudaq::ScalarOperator(function);
    //   auto reverse_order_op = scalar_op_1 * want_value_3;

    //   auto got_value = new_scalar_op.evaluate({want_value_1});
    //   auto got_value_1 = reverse_order_op.evaluate({want_value_1});

    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_3 * want_value_2),
    //               1e-5);
    //   EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_2 * want_value_3),
    //               1e-5);
    // }

    // // / : Constant scalar operator.
    // {
    //   auto scalar_op = cudaq::ScalarOperator(want_value_2);
    //   auto new_scalar_op = want_value_3 / scalar_op;

    //   auto scalar_op_1 = cudaq::ScalarOperator(want_value_2);
    //   auto reverse_order_op = scalar_op_1 / want_value_3;

    //   auto got_value = new_scalar_op.evaluate({});
    //   auto got_value_1 = reverse_order_op.evaluate({});

    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_3 / want_value_2),
    //               1e-5);
    //   EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_2 / want_value_3),
    //               1e-5);
    // }

    // // / : Scalar operator from lambda.
    // {
    //   auto function = [](std::vector<std::complex<double>> vec) {
    //     return vec[0];
    //   };
    //   auto scalar_op = cudaq::ScalarOperator(function);
    //   auto new_scalar_op = want_value_3 / scalar_op;

    //   auto scalar_op_1 = cudaq::ScalarOperator(function);
    //   auto reverse_order_op = scalar_op_1 / want_value_3;

    //   auto got_value = new_scalar_op.evaluate({want_value_1});
    //   auto got_value_1 = reverse_order_op.evaluate({want_value_1});

    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_3 / want_value_2),
    //               1e-5);
    //   EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value_2 / want_value_3),
    //               1e-5);
    // }

    // // += : Constant scalar operator.
    // {
    //   auto scalar_op = cudaq::ScalarOperator(want_value_0);
    //   scalar_op += want_value_0;

    //   auto got_value = scalar_op.evaluate({});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 + want_value_0),
    //               1e-5);
    // }

    // // += : Scalar operator from lambda.
    // {
    //   auto function = [](std::vector<std::complex<double>> vec) {
    //     return vec[0];
    //   };
    //   auto scalar_op = cudaq::ScalarOperator(function);
    //   scalar_op += want_value_1;

    //   auto got_value = scalar_op.evaluate({want_value_0});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 + want_value_1),
    //               1e-5);
    // }

    // // -= : Constant scalar operator.
    // {
    //   auto scalar_op = cudaq::ScalarOperator(want_value_0);
    //   scalar_op -= want_value_0;

    //   auto got_value = scalar_op.evaluate({});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 - want_value_0),
    //               1e-5);
    // }

    // // -= : Scalar operator from lambda.
    // {
    //   auto function = [](std::vector<std::complex<double>> vec) {
    //     return vec[0];
    //   };
    //   auto scalar_op = cudaq::ScalarOperator(function);
    //   scalar_op -= want_value_1;

    //   auto got_value = scalar_op.evaluate({want_value_0});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_0 - want_value_1),
    //               1e-5);
    // }

    // // *= : Constant scalar operator.
    // {
    //   auto scalar_op = cudaq::ScalarOperator(want_value_2);
    //   scalar_op *= want_value_3;

    //   auto got_value = scalar_op.evaluate({});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_2 * want_value_3),
    //               1e-5);
    // }

    // // *= : Scalar operator from lambda.
    // {
    //   auto function = [](std::vector<std::complex<double>> vec) {
    //     return vec[0];
    //   };
    //   auto scalar_op = cudaq::ScalarOperator(function);
    //   scalar_op *= want_value_3;

    //   auto got_value = scalar_op.evaluate({want_value_2});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_2 * want_value_3),
    //               1e-5);
    // }

    // // /= : Constant scalar operator.
    // {
    //   auto scalar_op = cudaq::ScalarOperator(want_value_2);
    //   scalar_op /= want_value_3;

    //   auto got_value = scalar_op.evaluate({});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_2 / want_value_3),
    //               1e-5);
    // }

    // // /= : Scalar operator from lambda.
    // {
    //   auto function = [](std::vector<std::complex<double>> vec) {
    //     return vec[0];
    //   };
    //   auto scalar_op = cudaq::ScalarOperator(function);
    //   scalar_op /= want_value_3;

    //   auto got_value = scalar_op.evaluate({want_value_2});
    //   EXPECT_NEAR(std::abs(got_value), std::abs(want_value_2 / want_value_3),
    //               1e-5);
    // }
  }
}