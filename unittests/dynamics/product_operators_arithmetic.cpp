/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <gtest/gtest.h>

#include <numeric>

namespace utils_1 {
void checkEqual(cudaq::matrix_2 a, cudaq::matrix_2 b) {
  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.get_rows(), b.get_rows());
  ASSERT_EQ(a.get_columns(), b.get_columns());
  ASSERT_EQ(a.get_size(), b.get_size());
  for (std::size_t i = 0; i < a.get_rows(); i++) {
    for (std::size_t j = 0; j < a.get_columns(); j++) {
      double a_val = a[{i, j}].real();
      double b_val = b[{i, j}].real();
      EXPECT_NEAR(a_val, b_val, 1e-8);
    }
  }
}

cudaq::matrix_2 zero_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  return mat;
}

cudaq::matrix_2 id_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = 1.0 + 0.0j;
  return mat;
}

cudaq::matrix_2 annihilate_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::matrix_2 create_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::matrix_2 position_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat[{i, i + 1}] = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::matrix_2 momentum_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] =
        (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat[{i, i + 1}] =
        -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::matrix_2 number_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = static_cast<double>(i) + 0.0j;
  return mat;
}

cudaq::matrix_2 parity_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
  return mat;
}

cudaq::matrix_2 displace_matrix(std::size_t size,
                                std::complex<double> amplitude) {
  auto term1 = amplitude * create_matrix(size);
  auto term2 = std::conj(amplitude) * annihilate_matrix(size);
  auto difference = term1 - term2;
  return difference.exponential();
}

cudaq::matrix_2 squeeze_matrix(std::size_t size,
                               std::complex<double> amplitude) {
  auto term1 = std::conj(amplitude) * annihilate_matrix(size).power(2);
  auto term2 = amplitude * create_matrix(size).power(2);
  auto difference = 0.5 * (term1 - term2);
  return difference.exponential();
}

} // namespace utils_1

TEST(ExpressionTester, checkSimple) {
  //   /// Sum of 3 elementary operators -- on the same DOF --
  //   /// then checking the sum matrix.
  //   {
  //   int level_count = 3;

  //   auto op0 = cudaq::elementary_operator::annihilate(0);
  //   auto op1 = cudaq::elementary_operator::create(0);
  //   auto op2 = cudaq::elementary_operator::identity(0);

  //   auto sum = op0 + op1 + op2;

  //   auto got_matrix = sum.to_matrix({{0, level_count}});
  //   std::cout << "\nterm 0 = " << op0.to_matrix({{0, level_count}}).dump() <<
  //   "\n"; std::cout << "\nterm 1 = " << op1.to_matrix({{0,
  //   level_count}}).dump() << "\n"; std::cout << "\nterm 2 = " <<
  //   op2.to_matrix({{0, level_count}}).dump() << "\n"; std::cout << "\ngot = "
  //   << got_matrix.dump() << "\n";

  // }

  //   /// Difference of 3 elementary operators -- on the same DOF --
  //   /// then checking the sum matrix.
  //   {
  //   int level_count = 3;

  //   auto op0 = cudaq::elementary_operator::annihilate(0);
  //   auto op1 = cudaq::elementary_operator::create(0);
  //   auto op2 = cudaq::elementary_operator::identity(0);

  //   auto difference = op0 - op1 - op2;

  //   auto got_matrix = difference.to_matrix({{0, level_count}});
  //   std::cout << "\nterm 0 = " << op0.to_matrix({{0, level_count}}).dump() <<
  //   "\n"; std::cout << "\nterm 1 = " << op1.to_matrix({{0,
  //   level_count}}).dump() << "\n"; std::cout << "\nterm 2 = " <<
  //   op2.to_matrix({{0, level_count}}).dump() << "\n"; std::cout << "\ngot = "
  //   << got_matrix.dump() << "\n";

  // }

  {
    auto op0 = cudaq::elementary_operator::annihilate(0);
    auto op1 = cudaq::elementary_operator::create(0);

    auto prod0 = cudaq::product_operator({}, {op0});
    auto prod1 = -1. * op1;
    // auto sum = cudaq::operator_sum({prod0, prod1});
    // // auto sum = prod0 + prod1;

    std::cout << "\n op0 = " << op0.to_matrix({{0, 2}}).dump() << "\n";
    std::cout << "\n op1 = " << op1.to_matrix({{0, 2}}).dump() << "\n";
    std::cout << "\n prod0 = " << prod0.to_matrix({{0, 2}}).dump() << "\n";
    std::cout << "\n prod1 = " << prod1.to_matrix({{0, 2}}).dump() << "\n";
    // std::cout << "\n sum = " << sum.to_matrix({{0,2}}).dump() << "\n";
  }
}

TEST(ExpressionTester, checkProductOperatorSimpleMatrixChecks) {
  std::vector<int> levels = {2, 3, 4};

  {
    //   // Same degrees of freedom.
    //   {
    //     for (auto level_count : levels) {
    //       auto op0 = cudaq::elementary_operator::annihilate(0);
    //       auto op1 = cudaq::elementary_operator::create(0);

    //       // std::cout << "\nannihilate got = \n" << op0.to_matrix({{0,
    //       level_count}}).dump() << "\n";
    //       // std::cout << "\ncreate got = \n" << op1.to_matrix({{0,
    //       level_count}}).dump() << "\n"; cudaq::product_operator got = op0 *
    //       op1;

    //       /// Check the matrices.
    //       /// FIXME: Comment me back in when `to_matrix` is implemented.

    //       auto got_matrix = got.to_matrix({{0, level_count}});
    //       auto matrix0 = utils_1::annihilate_matrix(level_count);
    //       auto matrix1 = utils_1::create_matrix(level_count);

    //       std::cout << "\nannihilate want = \n" << matrix0.dump() << "\n";
    //       std::cout << "\ncreate want = \n" << matrix1.dump() << "\n";
    //       auto want_matrix = matrix0 * matrix1;

    //       std::cout << "\ngot_matrix = \n" << got_matrix.dump() << "\n";
    //       std::cout << "\nwant_matrix = \n" << want_matrix.dump() << "\n";
    //       utils_1::checkEqual(want_matrix, got_matrix);

    //       std::vector<int> want_degrees = {0};
    //       ASSERT_TRUE(got.degrees() == want_degrees);
    //     }
    //   }

    // // Different degrees of freedom.
    // {
    //   for (auto level_count : levels) {
    //     auto op0 = cudaq::elementary_operator::annihilate(0);
    //     auto op1 = cudaq::elementary_operator::create(1);

    //     cudaq::product_operator got = op0 * op1;
    //     cudaq::product_operator got_reverse = op1 * op0;

    //     std::vector<int> want_degrees = {0, 1};
    //     ASSERT_TRUE(got.degrees() == want_degrees);
    //     ASSERT_TRUE(got_reverse.degrees() == want_degrees);

    //     /// Check the matrices.
    //     /// FIXME: Comment me back in when `to_matrix` is implemented.

    //     auto got_matrix =
    //         got.to_matrix({{0, level_count}, {1, level_count}}, {});
    //     auto got_matrix_reverse =
    //         got_reverse.to_matrix({{0, level_count}, {1, level_count}}, {});

    //     auto identity = utils_1::id_matrix(level_count);
    //     auto matrix0 = utils_1::annihilate_matrix(level_count);
    //     auto matrix1 = utils_1::create_matrix(level_count);

    //     auto fullHilbert0 = cudaq::kronecker(identity, matrix0);
    //     auto fullHilbert1 = cudaq::kronecker(matrix1, identity);
    //     auto want_matrix = fullHilbert0 * fullHilbert1;
    //     auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

    //     utils_1::checkEqual(want_matrix, got_matrix);
    //     utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
    //   }
    // }

    // Different degrees of freedom, non-consecutive.
    // Should produce the same matrices as the above test, as we don't
    // provide levels for the "missing" degree of freedom.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(2);

        cudaq::product_operator got = op0 * op1;
        cudaq::product_operator got_reverse = op1 * op0;

        std::vector<int> want_degrees = {0, 2};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);

        /// Check the matrices.
        /// FIXME: Comment me back in when `to_matrix` is implemented.

        // auto got_matrix = got.to_matrix({{0,level_count},{2,level_count}},
        // {});
        // auto got_matrix_reverse =
        // got_reverse.to_matrix({{0,level_count},{2,level_count}},
        // {});

        auto identity = utils_1::id_matrix(level_count);
        auto matrix0 = utils_1::annihilate_matrix(level_count);
        auto matrix1 = utils_1::create_matrix(level_count);

        auto fullHilbert0 = cudaq::kronecker(identity, matrix0);
        auto fullHilbert1 = cudaq::kronecker(matrix1, identity);
        auto want_matrix = fullHilbert0 * fullHilbert1;
        auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

        // utils_1::checkEqual(want_matrix, got_matrix);
        // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
      }
    }

    // Different degrees of freedom, non-consecutive but all dimensions
    // provided.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(2);

        cudaq::product_operator got = op0 * op1;
        cudaq::product_operator got_reverse = op1 * op0;

        std::vector<int> want_degrees = {0, 2};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);

        /// Check the matrices.
        /// FIXME: Comment me back in when `to_matrix` is implemented.

        // auto got_matrix =
        // got.to_matrix({{0,level_count},{1,level_count},{2,level_count}}, {});
        // auto got_matrix_reverse =
        // got_reverse.to_matrix({{0,level_count},{1,level_count},{2,level_count}},
        // {});

        auto identity = utils_1::id_matrix(level_count);
        auto matrix0 = utils_1::annihilate_matrix(level_count);
        auto matrix1 = utils_1::create_matrix(level_count);

        /// Identity pad the operators to compute the kronecker
        /// product to the full hilbert space.
        std::vector<cudaq::matrix_2> matrices_0;
        std::vector<cudaq::matrix_2> matrices_1;
        matrices_0 = {identity, identity, matrix0};
        matrices_1 = {matrix1, identity, identity};

        auto fullHilbert0 =
            cudaq::kronecker(matrices_0.begin(), matrices_0.end());
        auto fullHilbert1 =
            cudaq::kronecker(matrices_1.begin(), matrices_1.end());
        auto want_matrix = fullHilbert0 * fullHilbert1;
        auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

        // utils_1::checkEqual(want_matrix, got_matrix);
        // utils_1::checkEqual(got_matrix, want_matrix);
      }
    }
  }
}

TEST(ExpressionTester, checkProductOperatorAgainstScalars) {
  std::complex<double> value_0 = 0.1 + 0.1;
  int level_count = 3;

  /// `product_operator + complex<double>`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto sum = value_0 + product_op;
    auto reverse = product_op + value_0;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // sum.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::annihilate_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator + double`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto sum = 2.0 + product_op;
    auto reverse = product_op + 2.0;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // sum.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::annihilate_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity = 2.0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator + scalar_operator`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto sum = scalar_op + product_op;
    auto reverse = product_op + scalar_op;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // sum.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::annihilate_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator - complex<double>`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto difference = value_0 - product_op;
    auto reverse = product_op - value_0;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // difference.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::annihilate_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator - double`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto difference = 2.0 - product_op;
    auto reverse = product_op - 2.0;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // difference.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::annihilate_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity = 2.0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator - scalar_operator`
  {
    auto product_op = cudaq::elementary_operator::momentum(0) *
                      cudaq::elementary_operator::momentum(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto difference = scalar_op - product_op;
    auto reverse = product_op - scalar_op;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // difference.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::momentum_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::momentum_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator * complex<double>`
  {
    auto product_op = cudaq::elementary_operator::number(0) *
                      cudaq::elementary_operator::number(1);

    auto product = value_0 * product_op;
    auto reverse = product_op * value_0;

    ASSERT_TRUE(product.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::number_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator * double`
  {
    auto product_op = cudaq::elementary_operator::parity(0) *
                      cudaq::elementary_operator::parity(1);

    auto product = 2.0 * product_op;
    auto reverse = product_op * 2.0;

    ASSERT_TRUE(product.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse_reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::parity_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::parity_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 2.0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator * scalar_operator`
  {
    auto product_op = cudaq::elementary_operator::position(0) *
                      cudaq::elementary_operator::position(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto product = scalar_op * product_op;
    auto reverse = product_op * scalar_op;

    ASSERT_TRUE(product.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::position_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator *= complex<double>`
  {
    auto product = cudaq::elementary_operator::number(0) *
                   cudaq::elementary_operator::momentum(1);
    product *= value_0;

    ASSERT_TRUE(product.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::momentum_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator *= double`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::create(1);
    product *= 2.0;

    ASSERT_TRUE(product.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::annihilate_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::create_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 2.0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator *= scalar_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    product *= scalar_op;

    ASSERT_TRUE(product.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});

    auto term_0 = cudaq::kronecker(utils_1::id_matrix(level_count),
                                   utils_1::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils_1::momentum_matrix(level_count),
                                   utils_1::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils_1::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    // utils_1::checkEqual(want_matrix, got_matrix);
  }
}

TEST(ExpressionTester, checkProductOperatorAgainstProduct) {

  int level_count = 3;

  // `product_operator + product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::annihilate(2);

    auto sum = term_0 + term_1;

    ASSERT_TRUE(sum.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // sum.to_matrix({{0,level_count},{1,level_count}, {2,level_count+1}}, {});

    // Build up each individual term, cast to the full Hilbert space of the
    // system.
    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    matrices_0_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::annihilate_matrix(level_count)};
    matrices_0_1 = {utils_1::id_matrix(level_count + 1),
                    utils_1::annihilate_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    matrices_1_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::create_matrix(level_count),
                    utils_1::id_matrix(level_count)};
    matrices_1_1 = {utils_1::annihilate_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());

    auto want_matrix = term_0_matrix + term_1_matrix;
    // utils_1::checkEqual(want_matrix, got_matrix);
  }

  // `product_operator - product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::number(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::momentum(2);

    auto difference = term_0 - term_1;

    ASSERT_TRUE(difference.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // difference.to_matrix({{0,level_count},{1,level_count},
    // {2,level_count+1}}, {});

    // Build up each individual term, cast to the full Hilbert space of the
    // system.
    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    matrices_0_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::annihilate_matrix(level_count)};
    matrices_0_1 = {utils_1::id_matrix(level_count + 1),
                    utils_1::number_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    matrices_1_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::create_matrix(level_count),
                    utils_1::id_matrix(level_count)};
    matrices_1_1 = {utils_1::momentum_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());

    auto want_matrix = term_0_matrix - term_1_matrix;
    // utils_1::checkEqual(want_matrix, got_matrix);
  }

  // `product_operator * product_operator`
  {
    auto term_0 = cudaq::elementary_operator::position(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::parity(2);

    auto product = term_0 * term_1;

    ASSERT_TRUE(product.term_count() == 4);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}, {2,level_count+1}},
    // {});

    // Build up each individual term, cast to the full Hilbert space of the
    // system.
    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    matrices_0_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::position_matrix(level_count)};
    matrices_0_1 = {utils_1::id_matrix(level_count + 1),
                    utils_1::annihilate_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    matrices_1_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::create_matrix(level_count),
                    utils_1::id_matrix(level_count)};
    matrices_1_1 = {utils_1::parity_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());

    auto want_matrix = term_0_matrix * term_1_matrix;
    // utils_1::checkEqual(want_matrix, got_matrix);
  }

  // `product_operator *= product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::number(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::annihilate(2);

    term_0 *= term_1;

    ASSERT_TRUE(term_0.term_count() == 4);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // term_0.to_matrix({{0,level_count},{1,level_count}, {2,level_count+1}},
    // {});

    // Build up each individual term, cast to the full Hilbert space of the
    // system.
    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    matrices_0_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::annihilate_matrix(level_count)};
    matrices_0_1 = {utils_1::id_matrix(level_count + 1),
                    utils_1::number_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    matrices_1_0 = {utils_1::id_matrix(level_count + 1),
                    utils_1::create_matrix(level_count),
                    utils_1::id_matrix(level_count)};
    matrices_1_1 = {utils_1::annihilate_matrix(level_count + 1),
                    utils_1::id_matrix(level_count),
                    utils_1::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());

    auto want_matrix = term_0_matrix * term_1_matrix;
    // utils_1::checkEqual(want_matrix, got_matrix);
  }
}

TEST(ExpressionTester, checkProductOperatorAgainstElementary) {

  int level_count = 3;

  // `product_operator + elementary_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    auto sum = product + elementary;
    auto reverse = elementary + product;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // sum.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto product_matrix =
        cudaq::kronecker(utils_1::id_matrix(level_count),
                         utils_1::annihilate_matrix(level_count)) *
        cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                         utils_1::id_matrix(level_count));
    auto elementary_matrix = cudaq::kronecker(
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count));

    auto want_matrix = product_matrix + elementary_matrix;
    auto want_matrix_reverse = elementary_matrix + product_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator - elementary_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    auto difference = product - elementary;
    auto reverse = elementary - product;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // difference.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto product_matrix =
        cudaq::kronecker(utils_1::id_matrix(level_count),
                         utils_1::annihilate_matrix(level_count)) *
        cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                         utils_1::id_matrix(level_count));
    auto elementary_matrix = cudaq::kronecker(
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count));

    auto want_matrix = product_matrix - elementary_matrix;
    auto want_matrix_reverse = elementary_matrix - product_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator * elementary_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    auto product = term_0 * elementary;
    auto reverse = elementary * term_0;

    ASSERT_TRUE(product.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count}}, {});

    auto product_matrix =
        cudaq::kronecker(utils_1::id_matrix(level_count),
                         utils_1::annihilate_matrix(level_count)) *
        cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                         utils_1::id_matrix(level_count));
    auto elementary_matrix = cudaq::kronecker(
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count));

    auto want_matrix = product_matrix * elementary_matrix;
    auto want_matrix_reverse = elementary_matrix * product_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator *= elementary_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    product *= elementary;

    ASSERT_TRUE(product.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count}}, {});

    auto product_matrix =
        cudaq::kronecker(utils_1::id_matrix(level_count),
                         utils_1::annihilate_matrix(level_count)) *
        cudaq::kronecker(utils_1::annihilate_matrix(level_count),
                         utils_1::id_matrix(level_count));
    auto elementary_matrix = cudaq::kronecker(
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count));

    auto want_matrix = product_matrix * elementary_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
  }
}

TEST(ExpressionTester, checkProductOperatorAgainstOperatorSum) {

  int level_count = 3;

  // `product_operator + operator_sum`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto original_sum = cudaq::elementary_operator::create(1) +
                        cudaq::elementary_operator::create(2);

    auto sum = product + original_sum;
    auto reverse = original_sum + product;

    ASSERT_TRUE(sum.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // sum.to_matrix({{0,level_count},{1,level_count},{2,level_count+1}}, {});
    // auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count},{2,level_count+1}},
    // {});

    // Cast every term to full Hilbert space.
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils_1::id_matrix(level_count + 1), utils_1::id_matrix(level_count),
        utils_1::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils_1::id_matrix(level_count + 1),
        utils_1::annihilate_matrix(level_count),
        utils_1::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils_1::id_matrix(level_count + 1),
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils_1::create_matrix(level_count + 1),
        utils_1::id_matrix(level_count), utils_1::id_matrix(level_count)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix + sum_matrix;
    auto want_matrix_reverse = sum_matrix + product_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator - operator_sum`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto original_difference = cudaq::elementary_operator::create(1) -
                               cudaq::elementary_operator::create(2);

    auto difference = product - original_difference;
    auto reverse = original_difference - product;

    ASSERT_TRUE(difference.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // difference.to_matrix({{0,level_count},{1,level_count},{2,level_count+1}},
    // {}); auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count},{2,level_count+1}},
    // {});

    // Cast every term to full Hilbert space.
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils_1::id_matrix(level_count + 1), utils_1::id_matrix(level_count),
        utils_1::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils_1::id_matrix(level_count + 1),
        utils_1::annihilate_matrix(level_count),
        utils_1::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils_1::id_matrix(level_count + 1),
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils_1::create_matrix(level_count + 1),
        utils_1::id_matrix(level_count), utils_1::id_matrix(level_count)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto difference_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) -
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix - difference_matrix;
    auto want_matrix_reverse = difference_matrix - product_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator * operator_sum`
  {
    auto original_product = cudaq::elementary_operator::annihilate(0) *
                            cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = original_product * sum;
    auto reverse = sum * original_product;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix =
    // product.to_matrix({{0,level_count},{1,level_count},{2,level_count+1}},
    // {}); auto got_matrix_reverse =
    // reverse.to_matrix({{0,level_count},{1,level_count},{2,level_count+1}},
    // {});

    // Cast every term to full Hilbert space.
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils_1::id_matrix(level_count + 1), utils_1::id_matrix(level_count),
        utils_1::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils_1::id_matrix(level_count + 1),
        utils_1::annihilate_matrix(level_count),
        utils_1::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils_1::id_matrix(level_count + 1),
        utils_1::create_matrix(level_count), utils_1::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils_1::create_matrix(level_count + 1),
        utils_1::id_matrix(level_count), utils_1::id_matrix(level_count)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix * sum_matrix;
    auto want_matrix_reverse = sum_matrix * product_matrix;

    // utils_1::checkEqual(want_matrix, got_matrix);
    // utils_1::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}
