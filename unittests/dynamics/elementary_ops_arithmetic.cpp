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


/// NOTE: Not yet testing any of the matrix conversions. Just testing
/// the attributes of the output data type coming from the arithmetic.
/// These tests should be built upon to do actual numeric checks once
/// the implementations are complete.


TEST(ExpressionTester, checkPreBuiltElementaryOpsScalars) {

  auto function = [](std::map<std::string, std::complex<double>> parameters) {
    return parameters["value"];
  };

  // Addition against constant scalar operator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto sum = self + other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Addition against scalar operator from generator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self + other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Subtraction against constant scalar operator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto sum = self - other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Subtraction against scalar operator from generator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self - other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Multiplication against constant scalar operator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto product = self * other;
    ASSERT_TRUE(product.get_terms().size() == 2);
  }

  // Multiplication against scalar operator from generator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    ASSERT_TRUE(product.get_terms().size() == 2);
  }

  // Division against constant scalar operator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto product = self / other;
    ASSERT_TRUE(product.get_terms().size() == 2);
  }

  // Division against scalar operator from generator.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self / other;
    ASSERT_TRUE(product.get_terms().size() == 2);
  }

}

/// Prebuilt elementary ops against one another.
TEST(ExpressionTester, checkPreBuiltElementaryOpsSelf) {

  /// TODO: Check the output degrees attribute.

  // Addition, same DOF.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(1);

    auto sum = self + other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Subtraction, same DOF.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(1);

    auto sum = self - other;
    ASSERT_TRUE(sum.get_terms().size() == 2);
  }

  // Multiplication, same DOF.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(0);

    auto product = self * other;
    ASSERT_TRUE(product.get_terms().size() == 2);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(1);

    auto product = self * other;
    ASSERT_TRUE(product.get_terms().size() == 2);
  }

}