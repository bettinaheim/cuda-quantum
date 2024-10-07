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

TEST(ExpressionTester, checkScalarAgainstElementary) {

  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  auto local_variable = true;
  auto function = [&](std::map<std::string, std::complex<double>> parameters) {
    if (!local_variable)
      throw std::runtime_error("Local variable not detected.");
    return parameters["value"];
  };

  // Identity against constant.
  {
    auto id_op = cudaq::elementary_operator::identity(0);
    auto scalar_op = cudaq::scalar_operator(
        value_0); // cudaq::variable() , cudaq::operator::parameter() ,

    // auto multiplication = scalar_op * id_op;
    // auto addition = scalar_op + id_op;
    // auto subtraction = scalar_op - id_op;
  }

  // Identity against constant from lambda.
  {
    auto id_op = cudaq::elementary_operator::identity(0);
    auto scalar_op = cudaq::scalar_operator(function);

    // auto multiplication = scalar_op * id_op;
    // auto addition = scalar_op + id_op;
    // auto subtraction = scalar_op - id_op;
  }
}