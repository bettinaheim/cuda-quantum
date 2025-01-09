/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/matrix.h"
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

// cudaq::matrix_2 displace_matrix(std::size_t size,
//                                       std::complex<double> amplitude) {
//   auto mat = cudaq::matrix_2(size, size);
//   for (std::size_t i = 0; i + 1 < size; i++) {
//     mat[{i + 1, i}] =
//         amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
//     mat[{i, i + 1}] = -1. * std::conj(amplitude) * (0.5 * 'j') *
//                         std::sqrt(static_cast<double>(i + 1)) +
//                     0.0 * 'j';
//   }
//   return mat.exp();
// }

} // namespace utils_2

TEST(ExpressionTester, checkOperatorSumAgainstScalarOperator) {

  // `operator_sum * scalar_operator` and `scalar_operator * operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = sum * cudaq::scalar_operator(1.0);
    auto reverse = cudaq::scalar_operator(1.0) * sum;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }
  }

  // `operator_sum + scalar_operator` and `scalar_operator + operator_sum`
  {
    auto original = cudaq::elementary_operator::create(1) +
                    cudaq::elementary_operator::create(2);

    auto sum = original + cudaq::scalar_operator(1.0);
    auto reverse = cudaq::scalar_operator(1.0) + original;

    ASSERT_TRUE(sum.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum - scalar_operator` and `scalar_operator - operator_sum`
  {
    auto original = cudaq::elementary_operator::create(1) +
                    cudaq::elementary_operator::create(2);

    auto difference = original - cudaq::scalar_operator(1.0);
    auto reverse = cudaq::scalar_operator(1.0) - original;

    ASSERT_TRUE(difference.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum *= scalar_operator`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum *= cudaq::scalar_operator(1.0);

    ASSERT_TRUE(sum.term_count() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }
  }

  // `operator_sum += scalar_operator`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum += cudaq::scalar_operator(1.0);

    ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= scalar_operator`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum -= cudaq::scalar_operator(1.0);

    ASSERT_TRUE(sum.term_count() == 3);
  }
}

TEST(ExpressionTester, checkOperatorSumAgainstScalars) {
  std::complex<double> value = 0.1 + 0.1;

  // `operator_sum * double` and `double * operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = sum * 2.0;
    auto reverse = 2.0 * sum;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }
  }

  // `operator_sum + double` and `double + operator_sum`
  {
    auto original = cudaq::elementary_operator::create(1) +
                    cudaq::elementary_operator::create(2);

    auto sum = original + 2.0;
    auto reverse = 2.0 + original;

    ASSERT_TRUE(sum.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum - double` and `double - operator_sum`
  {
    auto original = cudaq::elementary_operator::create(1) +
                    cudaq::elementary_operator::create(2);

    auto difference = original - 2.0;
    auto reverse = 2.0 - original;

    ASSERT_TRUE(difference.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum *= double`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum *= 2.0;

    ASSERT_TRUE(sum.term_count() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }
  }

  // `operator_sum += double`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum += 2.0;

    ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= double`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum -= 2.0;

    ASSERT_TRUE(sum.term_count() == 3);
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

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }
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
  }

  // `operator_sum *= std::complex<double>`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum *= value;

    ASSERT_TRUE(sum.term_count() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.term_count() == 2);
    }
  }

  // `operator_sum += std::complex<double>`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum += value;

    ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= std::complex<double>`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum -= value;

    ASSERT_TRUE(sum.term_count() == 3);
  }
}

TEST(ExpressionTester, checkOperatorSumAgainstOperatorSum) {
  // `operator_sum + operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) +
                 cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::identity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(3);

    auto sum = sum_0 + sum_1;

    ASSERT_TRUE(sum.term_count() == 5);
  }

  // `operator_sum - operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) +
                 cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::identity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(2);

    auto difference = sum_0 - sum_1;

    ASSERT_TRUE(difference.term_count() == 5);
  }

  // `operator_sum * operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) +
                 cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::identity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(2);

    auto sum_product = sum_0 * sum_1;

    ASSERT_TRUE(sum_product.term_count() == 6);
    for (auto term : sum_product.get_terms())
      ASSERT_TRUE(term.term_count() == 2);
  }

  // `operator_sum *= operator_sum`
  {
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::identity(0) +
                 cudaq::elementary_operator::annihilate(1) +
                 cudaq::elementary_operator::create(2);

    sum *= sum_1;

    ASSERT_TRUE(sum.term_count() == 6);
    for (auto term : sum.get_terms())
      ASSERT_TRUE(term.term_count() == 2);
  }
}

/// NOTE: Much of the simpler arithmetic between the two is tested in the
/// product operator test file. This mainly just tests the assignment operators
/// between the two types.
TEST(ExpressionTester, checkOperatorSumAgainstProduct) {
  // `operator_sum += product_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum += product;

    ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= product_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum -= product;

    ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum *= product_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    sum *= product;

    ASSERT_TRUE(sum.term_count() == 2);

    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.term_count() == 3);
    }
  }
}
