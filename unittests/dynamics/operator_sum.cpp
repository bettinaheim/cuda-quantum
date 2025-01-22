/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <gtest/gtest.h>

namespace utils_2 {
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

} // namespace utils_2

TEST(ExpressionTester, checkOperatorSumAgainstScalarOperator) {
  int level_count = 3;
  std::complex<double> value = 0.2 + 0.2j;

  // `operator_sum * scalar_operator` and `scalar_operator * operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = sum * cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) * sum;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    std::cout << "got matrix = " << got_matrix.dump() << "\n";
    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    utils_2::checkEqual(want_matrix, got_matrix);
    utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // // `operator_sum + scalar_operator` and `scalar_operator + operator_sum`
  // {
  //   level_count = 2;
  //   auto original = cudaq::elementary_operator::create(1) +
  //                   cudaq::elementary_operator::create(2);

  //   auto sum = original + cudaq::scalar_operator(value);
  //   auto reverse = cudaq::scalar_operator(value) + original;

  //   ASSERT_TRUE(sum.term_count() == 3);
  //   ASSERT_TRUE(reverse.term_count() == 3);

  //   /// Check the matrices.
  //   /// FIXME: Comment me back in when `to_matrix` is implemented.

  //   // Only providing dimensions for the `1` and `2` degrees of freedom.
  //   auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
  //   auto got_matrix_reverse = reverse.to_matrix({{1, level_count},
  //   {2,level_count+1}});

  //   auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
  //                                   utils_2::create_matrix(level_count));
  //   auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
  //                                   utils_2::id_matrix(level_count));
  //   auto sum_matrix = matrix0 + matrix1;
  //   auto scaled_identity =
  //       value * utils_2::id_matrix((level_count) * (level_count + 1));

  //   auto want_matrix = sum_matrix + scaled_identity;
  //   auto want_matrix_reverse = scaled_identity + sum_matrix;
  //   utils_2::checkEqual(want_matrix, got_matrix);
  //   utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  // }

  // // `operator_sum - scalar_operator` and `scalar_operator - operator_sum`
  // {
  //   auto original = cudaq::elementary_operator::create(1) +
  //                   cudaq::elementary_operator::create(2);

  //   auto difference = original - cudaq::scalar_operator(value);
  //   auto reverse = cudaq::scalar_operator(value) - original;

  //   ASSERT_TRUE(difference.term_count() == 3);
  //   ASSERT_TRUE(reverse.term_count() == 3);

  //   /// Check the matrices.
  //   /// FIXME: Comment me back in when `to_matrix` is implemented.

  //   // Only providing dimensions for the `1` and `2` degrees of freedom.
  //   // auto got_matrix = difference.to_matrix({{1, level_count}, {2,
  //   // level_count+1}}); auto got_matrix_reverse = reverse.to_matrix({{1,
  //   // level_count}, {2, level_count+1}});

  //   auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
  //                                   utils_2::create_matrix(level_count));
  //   auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
  //                                   utils_2::id_matrix(level_count));
  //   auto sum_matrix = matrix0 + matrix1;
  //   auto scaled_identity =
  //       value * utils_2::id_matrix((level_count) * (level_count + 1));

  //   auto want_matrix = sum_matrix - scaled_identity;
  //   auto want_matrix_reverse = scaled_identity - sum_matrix;
  //   // utils_2::checkEqual(want_matrix, got_matrix);
  //   // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  // }

  // // `operator_sum *= scalar_operator`
  // {
  //   auto sum = cudaq::elementary_operator::create(1) +
  //              cudaq::elementary_operator::momentum(2);

  //   sum *= cudaq::scalar_operator(value);

  //   ASSERT_TRUE(sum.term_count() == 2);

  //   /// Check the matrices.
  //   /// FIXME: Comment me back in when `to_matrix` is implemented.

  //   // Providing dimensions for the `0`, `1` and `2` degrees of freedom.
  //   // auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count},
  //   {2,
  //   // level_count+1}});

  //   std::vector<cudaq::matrix_2> matrices_1 = {
  //       utils_2::id_matrix(level_count + 1),
  //       utils_2::create_matrix(level_count),
  //       utils_2::id_matrix(level_count)};
  //   std::vector<cudaq::matrix_2> matrices_2 = {
  //       utils_2::momentum_matrix(level_count + 1),
  //       utils_2::id_matrix(level_count), utils_2::id_matrix(level_count)};
  //   auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  //   auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
  //   auto scaled_identity =
  //       value *
  //       utils_2::id_matrix((level_count + 1) * level_count * level_count);

  //   auto want_matrix = (matrix0 + matrix1) * scaled_identity;
  //   // utils_2::checkEqual(want_matrix, got_matrix);
  // }

  // // `operator_sum += scalar_operator`
  // {
  //   auto sum = cudaq::elementary_operator::parity(1) +
  //              cudaq::elementary_operator::position(2);

  //   sum += cudaq::scalar_operator(value);

  //   ASSERT_TRUE(sum.term_count() == 3);

  //   /// Check the matrices.
  //   /// FIXME: Comment me back in when `to_matrix` is implemented.

  //   // Providing dimensions for the `0`, `1` and `2` degrees of freedom.
  //   // auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count},
  //   {2,
  //   // level_count+1}});

  //   std::vector<cudaq::matrix_2> matrices_1 = {
  //       utils_2::id_matrix(level_count + 1),
  //       utils_2::parity_matrix(level_count),
  //       utils_2::id_matrix(level_count)};
  //   std::vector<cudaq::matrix_2> matrices_2 = {
  //       utils_2::position_matrix(level_count + 1),
  //       utils_2::id_matrix(level_count), utils_2::id_matrix(level_count)};
  //   auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  //   auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
  //   auto scaled_identity =
  //       value *
  //       utils_2::id_matrix((level_count + 1) * level_count * level_count);

  //   auto want_matrix = matrix0 + matrix1 + scaled_identity;
  //   // utils_2::checkEqual(want_matrix, got_matrix);
  // }

  // // `operator_sum -= scalar_operator`
  // {
  //   auto sum = cudaq::elementary_operator::number(1) +
  //              cudaq::elementary_operator::annihilate(2);

  //   sum -= cudaq::scalar_operator(value);

  //   ASSERT_TRUE(sum.term_count() == 3);

  //   /// Check the matrices.
  //   /// FIXME: Comment me back in when `to_matrix` is implemented.

  //   // Providing dimensions for the `0`, `1` and `2` degrees of freedom.
  //   // auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count},
  //   {2,
  //   // level_count+1}});

  //   std::vector<cudaq::matrix_2> matrices_1 = {
  //       utils_2::id_matrix(level_count + 1),
  //       utils_2::number_matrix(level_count),
  //       utils_2::id_matrix(level_count)};
  //   std::vector<cudaq::matrix_2> matrices_2 = {
  //       utils_2::annihilate_matrix(level_count + 1),
  //       utils_2::id_matrix(level_count), utils_2::id_matrix(level_count)};
  //   auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  //   auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
  //   auto scaled_identity =
  //       value *
  //       utils_2::id_matrix((level_count + 1) * level_count * level_count);

  //   auto want_matrix = (matrix0 + matrix1) - scaled_identity;
  //   // utils_2::checkEqual(want_matrix, got_matrix);
  // }
}

TEST(ExpressionTester, checkOperatorSumAgainstScalars) {
  int level_count = 3;
  std::complex<double> value = 0.1 + 0.1j;
  double double_value = 0.1;

  // `operator_sum * double` and `double * operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = sum * double_value;
    auto reverse = double_value * sum;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = product.to_matrix({{1, level_count}, {2,
    // level_count+1}},
    // {}); auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2,
    // level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum + double` and `double + operator_sum`
  {
    auto original = cudaq::elementary_operator::momentum(1) +
                    cudaq::elementary_operator::position(2);

    auto sum = original + double_value;
    auto reverse = double_value + original;

    ASSERT_TRUE(sum.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
    // auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2,
    // level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::momentum_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::position_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    auto want_matrix_reverse = scaled_identity + sum_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - double` and `double - operator_sum`
  {
    auto original = cudaq::elementary_operator::parity(1) +
                    cudaq::elementary_operator::number(2);

    auto difference = original - double_value;
    auto reverse = double_value - original;

    ASSERT_TRUE(difference.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
    // auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2,
    // level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::number_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= double`
  {
    auto sum = cudaq::elementary_operator::squeeze(1) +
               cudaq::elementary_operator::squeeze(2);

    sum *= double_value;

    ASSERT_TRUE(sum.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}},
    // {{"squeezing", value}});

    auto matrix0 =
        cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                         utils_2::squeeze_matrix(level_count, value));
    auto matrix1 =
        cudaq::kronecker(utils_2::squeeze_matrix(level_count + 1, value),
                         utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += double`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum += double_value;

    ASSERT_TRUE(sum.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));

    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= double`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum -= double_value;

    ASSERT_TRUE(sum.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));

    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum * std::complex<double>` and `std::complex<double> *
  // operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = sum * value;
    auto reverse = value * sum;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = product.to_matrix({{1,level_count}, {2,
    // level_count+1}}); auto got_matrix_reverse =
    // reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum + std::complex<double>` and `std::complex<double> +
  // operator_sum`
  {
    auto original = cudaq::elementary_operator::create(1) +
                    cudaq::elementary_operator::create(2);

    auto sum = original + value;
    auto reverse = value + original;

    ASSERT_TRUE(sum.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});
    // auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2,
    // level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    auto want_matrix_reverse = scaled_identity + sum_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - std::complex<double>` and `std::complex<double> -
  // operator_sum`
  {
    auto original = cudaq::elementary_operator::create(1) +
                    cudaq::elementary_operator::create(2);

    auto difference = original - value;
    auto reverse = value - original;

    ASSERT_TRUE(difference.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = difference.to_matrix({{1,level_count}, {2,
    // level_count+1}}); auto got_matrix_reverse =
    // reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::create_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= std::complex<double>`
  {
    auto sum = cudaq::elementary_operator::displace(1) +
               cudaq::elementary_operator::parity(2);

    sum *= value;

    ASSERT_TRUE(sum.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}},
    // {{"displacement", value}});

    auto matrix0 =
        cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                         utils_2::displace_matrix(level_count, value));
    auto matrix1 = cudaq::kronecker(utils_2::parity_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += std::complex<double>`
  {
    auto sum = cudaq::elementary_operator::momentum(1) +
               cudaq::elementary_operator::squeeze(2);

    sum += value;

    ASSERT_TRUE(sum.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}},
    // {{"squeezing", value}});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::momentum_matrix(level_count));
    auto matrix1 =
        cudaq::kronecker(utils_2::squeeze_matrix(level_count + 1, value),
                         utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= std::complex<double>`
  {
    auto sum = cudaq::elementary_operator::position(1) +
               cudaq::elementary_operator::number(2);

    sum -= value;

    ASSERT_TRUE(sum.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // Only providing dimensions for the `1` and `2` degrees of freedom.
    // auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}},
    // {});

    auto matrix0 = cudaq::kronecker(utils_2::id_matrix(level_count + 1),
                                    utils_2::position_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils_2::number_matrix(level_count + 1),
                                    utils_2::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils_2::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }
}

TEST(ExpressionTester, checkOperatorSumAgainstOperatorSum) {
  int level_count = 2;

  // `operator_sum + operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) +
                 cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::parity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(3);

    auto sum = sum_0 + sum_1;

    ASSERT_TRUE(sum.term_count() == 5);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = sum.to_matrix({{0,level_count}, {1, level_count+1}, {2,
    // level_count+2}, {3, level_count+3}}, {});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::create_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_0_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::create_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::parity_matrix(level_count)};
    matrices_1_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::annihilate_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_2 = {utils_2::create_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix + sum_1_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum - operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) +
                 cudaq::elementary_operator::position(2);
    auto sum_1 = cudaq::elementary_operator::parity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::momentum(3);

    auto difference = sum_0 - sum_1;

    ASSERT_TRUE(difference.term_count() == 5);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = difference.to_matrix({{0,level_count}, {1,
    // level_count+1}, {2, level_count+2}, {3, level_count+3}}, {});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::create_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_0_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::position_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::parity_matrix(level_count)};
    matrices_1_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::annihilate_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_2 = {utils_2::momentum_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix - sum_1_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum * operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) +
                 cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::parity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(3);

    auto sum_product = sum_0 * sum_1;
    auto sum_product_reverse = sum_1 * sum_0;

    ASSERT_TRUE(sum_product.term_count() == 6);
    ASSERT_TRUE(sum_product_reverse.term_count() == 6);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = sum_product.to_matrix({{0,level_count}, {1,
    // level_count+1}, {2, level_count+2}, {3, level_count+3}}, {}); auto
    // got_matrix_reverse = sum_product_reverse.to_matrix({{0,level_count}, {1,
    // level_count+1}, {2, level_count+2}, {3, level_count+3}}, {});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::create_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_0_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::create_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::parity_matrix(level_count)};
    matrices_1_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::annihilate_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_2 = {utils_2::create_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix * sum_1_matrix;
    auto want_matrix_reverse = sum_1_matrix * sum_0_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
    // utils_2::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::parity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(3);

    sum *= sum_1;

    ASSERT_TRUE(sum.term_count() == 6);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = sum.to_matrix({{0,level_count}, {1,
    // level_count+1}, {2, level_count+2}, {3, level_count+3}}, {});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::create_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_0_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::create_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_0 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::parity_matrix(level_count)};
    matrices_1_1 = {utils_2::id_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::annihilate_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};
    matrices_1_2 = {utils_2::create_matrix(level_count + 3),
                    utils_2::id_matrix(level_count + 2),
                    utils_2::id_matrix(level_count + 1),
                    utils_2::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix * sum_1_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }
}

/// NOTE: Much of the simpler arithmetic between the two is tested in the
/// product operator test file. This mainly just tests the assignment operators
/// between the two types.
TEST(ExpressionTester, checkOperatorSumAgainstProduct) {
  int level_count = 2;

  // `operator_sum += product_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum += product;

    ASSERT_TRUE(sum.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1},
    // {2, level_count+2}}, {});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::id_matrix(level_count + 1),
        utils_2::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::annihilate_matrix(level_count + 1),
        utils_2::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::create_matrix(level_count + 1),
        utils_2::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils_2::create_matrix(level_count + 2),
        utils_2::id_matrix(level_count + 1), utils_2::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix + product_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= product_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum -= product;

    ASSERT_TRUE(sum.term_count() == 3);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1},
    // {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::id_matrix(level_count + 1),
        utils_2::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::annihilate_matrix(level_count + 1),
        utils_2::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::create_matrix(level_count + 1),
        utils_2::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils_2::create_matrix(level_count + 2),
        utils_2::id_matrix(level_count + 1), utils_2::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix - product_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum *= product_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum *= product;

    ASSERT_TRUE(sum.term_count() == 2);

    /// Check the matrices.
    /// FIXME: Comment me back in when `to_matrix` is implemented.

    // auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1},
    // {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::id_matrix(level_count + 1),
        utils_2::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::annihilate_matrix(level_count + 1),
        utils_2::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils_2::id_matrix(level_count + 2),
        utils_2::create_matrix(level_count + 1),
        utils_2::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils_2::create_matrix(level_count + 2),
        utils_2::id_matrix(level_count + 1), utils_2::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix * product_matrix;
    // utils_2::checkEqual(want_matrix, got_matrix);
  }
}
