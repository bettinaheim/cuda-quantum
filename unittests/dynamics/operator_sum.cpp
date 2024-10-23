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

// cudaq::complex_matrix _id_matrix(int size) {
//   auto mat = cudaq::complex_matrix(size, size);
//   for (int i = 0; i < size; i++)
//     mat(i, i) = 1.0 + 0.0j;
//   return mat;
// }

// cudaq::complex_matrix _annihilate_matrix(int size) {
//   auto mat = cudaq::complex_matrix(size, size);
//   for (std::size_t i = 0; i + 1 < size; i++)
//     mat(i, i + 1) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
//   return mat;
// }

// cudaq::complex_matrix _create_matrix(int size) {
//   auto mat = cudaq::complex_matrix(size, size);
//   for (std::size_t i = 0; i + 1 < size; i++)
//     mat(i + 1, i) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
//   return mat;
// }

// TEST(ExpressionTester, checkProductOperatorSimple) {
//   std::vector<int> levels = {2, 3, 4};

//   // std::set<int> uniqueDegrees;
//   // std::copy(this->degrees.begin(), this->degrees.end(),
//   // std::inserter(uniqueDegrees, uniqueDegrees.begin()));
//   // std::copy(other.degrees.begin(), other.degrees.end(),
//   // std::inserter(uniqueDegrees, uniqueDegrees.begin()));

//   // Arithmetic only between elementary operators with
//   // same number of levels.
//   {
//     // Same degrees of freedom.
//     {
//       for (auto level_count : levels) {
//         auto op0 = cudaq::elementary_operator::annihilate(0);
//         auto op1 = cudaq::elementary_operator::create(0);

//         cudaq::product_operator got = op0 * op1;
//         auto got_matrix = got.to_matrix({{0, level_count}}, {});

//         auto matrix0 = _annihilate_matrix(level_count);
//         auto matrix1 = _create_matrix(level_count);
//         auto want_matrix = matrix0 * matrix1;

//         // ASSERT_TRUE(want_matrix == got_matrix);
//       }
//     }

//     // // Different degrees of freedom.
//     // {
//     //   for (auto level_count : levels) {
//     //     auto op0 = cudaq::elementary_operator::annihilate(0);
//     //     auto op1 = cudaq::elementary_operator::create(1);

//     //     cudaq::product_operator got = op0 * op1;
//     //     auto got_matrix =
//     //         got.to_matrix({{0, level_count}, {1, level_count}}, {});

//     //     cudaq::product_operator got_reverse = op1 * op0;
//     //     auto got_matrix_reverse =
//     //         got_reverse.to_matrix({{0, level_count}, {1, level_count}}, {});

//     //     auto identity = _id_matrix(level_count);
//     //     auto matrix0 = _annihilate_matrix(level_count);
//     //     auto matrix1 = _create_matrix(level_count);

//     //     auto fullHilbert0 = identity.kronecker(matrix0);
//     //     auto fullHilbert1 = matrix1.kronecker(identity);
//     //     auto want_matrix = fullHilbert0 * fullHilbert1;
//     //     auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

//     //     // ASSERT_TRUE(want_matrix == got_matrix);
//     //     // ASSERT_TRUE(want_matrix_reverse == got_matrix_reverse);
//     //   }
//     // }

//     // // Different degrees of freedom, non-consecutive.
//     // {
//     //   for (auto level_count : levels) {
//     //     auto op0 = cudaq::elementary_operator::annihilate(0);
//     //     auto op1 = cudaq::elementary_operator::create(2);

//     //     // cudaq::product_operator got = op0 * op1;
//     //     // auto got_matrix = got.to_matrix({{0,level_count},{2,level_count}},
//     //     // {});
//     //   }
//     // }

//     // // Different degrees of freedom, non-consecutive but all dimensions
//     // // provided.
//     // {
//     //   for (auto level_count : levels) {
//     //     auto op0 = cudaq::elementary_operator::annihilate(0);
//     //     auto op1 = cudaq::elementary_operator::create(2);

//     //     // cudaq::product_operator got = op0 * op1;
//     //     // auto got_matrix =
//     //     // got.to_matrix({{0,level_count},{1,level_count},{2,level_count}},
//     //     {});
//     //   }
//     // }
//   }
// }

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


TEST(ExpressionTester, checkTest) {
  auto op = cudaq::elementary_operator::create(1);
  std::vector<std::variant<cudaq::scalar_operator, cudaq::elementary_operator>> ops = {op};
  auto prod = cudaq::product_operator({op});
  // auto prod = cudaq::product_operator(ops);
  std::cout << "\nsize = " << prod.term_count() << "\n";
}


TEST(ExpressionTester, checkOperatorSumAgainstScalarOperator) {

  // `operator_sum * scalar_operator` and `scalar_operator * operator_sum`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

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
     auto original = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     auto sum = original + cudaq::scalar_operator(1.0);
     auto reverse = cudaq::scalar_operator(1.0) + original;

     ASSERT_TRUE(sum.term_count() == 3);
     ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum - scalar_operator` and `scalar_operator - operator_sum`
  {
     auto original = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     auto difference = original - cudaq::scalar_operator(1.0);
     auto reverse = cudaq::scalar_operator(1.0) - original;

     ASSERT_TRUE(difference.term_count() == 3);
     ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum *= scalar_operator`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum *= cudaq::scalar_operator(1.0);

     ASSERT_TRUE(sum.term_count() == 2);
      for (auto term : sum.get_terms()) {
        ASSERT_TRUE(term.term_count() == 2);
      }
  }

  // `operator_sum += scalar_operator`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum += cudaq::scalar_operator(1.0);

     ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= scalar_operator`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum -= cudaq::scalar_operator(1.0);

     ASSERT_TRUE(sum.term_count() == 3);
  }
}



TEST(ExpressionTester, checkOperatorSumAgainstScalars) {
  std::complex<double> value = 0.1 + 0.1;
  
  // `operator_sum * double` and `double * operator_sum`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

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
     auto original = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     auto sum = original + 2.0;
     auto reverse = 2.0 + original;

     ASSERT_TRUE(sum.term_count() == 3);
     ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum - double` and `double - operator_sum`
  {
     auto original = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     auto difference = original - 2.0;
     auto reverse = 2.0 - original;

     ASSERT_TRUE(difference.term_count() == 3);
     ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum *= double`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum *= 2.0;

     ASSERT_TRUE(sum.term_count() == 2);
      for (auto term : sum.get_terms()) {
        ASSERT_TRUE(term.term_count() == 2);
      }
  }

  // `operator_sum += double`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum += 2.0;

     ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= double`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum -= 2.0;

     ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum * std::complex<double>` and `std::complex<double> * operator_sum`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

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

  // `operator_sum + std::complex<double>` and `std::complex<double> + operator_sum`
  {
     auto original = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     auto sum = original + value;
     auto reverse = value + original;

     ASSERT_TRUE(sum.term_count() == 3);
     ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum - std::complex<double>` and `std::complex<double> - operator_sum`
  {
     auto original = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     auto difference = original - value;
     auto reverse = value - original;

     ASSERT_TRUE(difference.term_count() == 3);
     ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `operator_sum *= std::complex<double>`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum *= value;

     ASSERT_TRUE(sum.term_count() == 2);
      for (auto term : sum.get_terms()) {
        ASSERT_TRUE(term.term_count() == 2);
      }
  }

  // `operator_sum += std::complex<double>`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum += value;

     ASSERT_TRUE(sum.term_count() == 3);
  }

  // `operator_sum -= std::complex<double>`
  {
     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

     sum -= value;

     ASSERT_TRUE(sum.term_count() == 3);
  }
}





TEST(ExpressionTester, checkOperatorSumAgainstOperatorSum) {
  // `operator_sum + operator_sum`
  {
    auto sum_0 = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);
    auto sum_1 = cudaq::elementary_operator::identity(0) + cudaq::elementary_operator::annihilate(1);

    auto sum = sum_0 + sum_1;

    ASSERT_TRUE(sum.term_count() == 4);
  }

  // `operator_sum - operator_sum`
  {

  }

  // `operator_sum * operator_sum`
  {

  }

}

// /// NOTE: Much of the simpler arithmetic between the two is tested in the
// /// product operator test file. This mainly just tests the assignment operators
// /// between the two types.
// TEST(ExpressionTester, checkOperatorSumAgainstProduct) {
//   // `operator_sum += product_operator`
//   {
//     auto product = cudaq::elementary_operator::annihilate(0) * cudaq::elementary_operator::annihilate(1);
//     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

//     sum += product;

//     ASSERT_TRUE(sum.term_count() == 3);
//   }

//   /// FIXME:
//   // // `operator_sum -= product_operator`
//   // {
//     // auto product = cudaq::elementary_operator::annihilate(0) * cudaq::elementary_operator::annihilate(1);
//     // auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

//     // auto sum -= product;

//     // ASSERT_TRUE(sum.term_count() == 3);
//   // }

//   // `operator_sum *= product_operator`
//   {
//     auto product = cudaq::elementary_operator::annihilate(0) * cudaq::elementary_operator::annihilate(1);
//     auto sum = cudaq::elementary_operator::create(1) + cudaq::elementary_operator::create(2);

//     sum *= product;

//     ASSERT_TRUE(sum.term_count() == 2);

//     for (auto term : sum.get_terms()) {
//       ASSERT_TRUE(term.term_count() == 2);
//     }

//   }
// }