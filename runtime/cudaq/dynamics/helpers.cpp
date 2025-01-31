/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "helpers.h"

namespace cudaq {
namespace detail {

  std::vector<std::string>
  generate_all_states(std::vector<int> degrees, std::map<int, int> dimensions) {
    if (degrees.size() == 0)
      return {};

    std::vector<std::string> states;
    int range = dimensions[degrees[0]];
    for (auto state = 0; state < range; state++) {
      states.push_back(std::to_string(state));
    }

    for (auto degree = degrees.begin() + 1; degree != degrees.end(); ++degree) {
      std::string term;
      std::vector<std::string> result;
      for (auto current : states) {
        for (auto state = 0; state < dimensions[degrees[*degree]]; state++) {
          result.push_back(current + std::to_string(state));
        }
      }
      states = result;
    }

    return states;
  }

  cudaq::matrix_2 permute_matrix(cudaq::matrix_2 matrix,
                                        std::vector<int> permutation) {
    auto result = cudaq::matrix_2(matrix.get_rows(), matrix.get_columns());
    std::vector<std::complex<double>> sorted_values;
    for (std::size_t permuted : permutation) {
      for (std::size_t permuted_again : permutation) {
        sorted_values.push_back(matrix[{permuted, permuted_again}]);
      }
    }
    int idx = 0;
    for (std::size_t row = 0; row < result.get_rows(); row++) {
      for (std::size_t col = 0; col < result.get_columns(); col++) {
        result[{row, col}] = sorted_values[idx];
        idx++;
      }
    }
    return result;
  }

  std::vector<int> canonicalize_degrees(std::vector<int> degrees) {
    std::sort(degrees.begin(), degrees.end(), std::greater<int>());
    return degrees;
  }

}
}
