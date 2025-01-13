/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/manipulation.h"

#include <ranges>

class _OperatorHelpers {
public:
  _OperatorHelpers() = default;

  // Permutes the given matrix according to the given permutation.
  // If states is the current order of vector entries on which the given matrix
  // acts, and permuted_states is the desired order of an array on which the
  // permuted matrix should act, then the permutation is defined such that
  // [states[i] for i in permutation] produces permuted_states.
  cudaq::matrix_2 permute_matrix(cudaq::matrix_2 matrix, std::vector<std::size_t> permutation) {
    auto result = cudaq::matrix_2(matrix.get_rows(), matrix.get_columns());
    std::vector<std::complex<double>> sorted_values;
    for (std::size_t permuted : permutation) {
      for (std::size_t permuted_again : permutation) {
        sorted_values.push_back(matrix[{permuted, permuted_again}]);
      }
    }
    int idx = 0;
    for (std::size_t row=0; row < result.get_rows(); row++) {
      for (std::size_t col=0; col < result.get_columns(); col++) {
        result[{row,col}] = sorted_values[idx];
        idx++;
      }
    }
    return result;
  }

};
