/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <memory>
#include <optional>
#include "cudaq/qis/state.h"
#include "common/ObserveResult.h"

namespace cudaq {

/// @brief The evolve_result encapsulates all data generated from a
/// cudaq"::"evolve call. This includes information about the state
/// and any computed expectation values during and after evolution,
/// depending on the arguments passed to the call.
class evolve_result {
protected:
  // The final state after the time evolution.
  state final_state;

  // The state after each step in the evolution, if computed.
  std::optional<std::vector<state>> intermediate_states = {};

  // The computed expectation values at the end of the evolution.
  std::optional<std::vector<observe_result>> final_expectation_values = {};

  // The computed expectation values for each step of the evolution.
  std::optional<std::vector<std::vector<observe_result>>> expectation_values = {};

public:

  evolve_result(state state)
    : final_state(state) {}

  evolve_result(state state, std::vector<observe_result> expectations)
    : final_state(state), 
      final_expectation_values(std::make_optional<std::vector<observe_result>>(expectations)) {}

  evolve_result(std::vector<state> states) 
    : final_state(states.back()), 
      intermediate_states(std::make_optional<std::vector<state>>(states)) {}

  evolve_result(std::vector<state> states, 
                std::vector<std::vector<observe_result>> expectations) 
    : final_state(states.back()),
      intermediate_states(std::make_optional<std::vector<state>>(states)), 
      final_expectation_values(std::make_optional<std::vector<observe_result>>(expectations.back())),
      expectation_values(std::make_optional<std::vector<std::vector<observe_result>>>(expectations)) {}

  state get_final_state() { 
    return final_state; 
  }

  std::optional<std::vector<state>> get_intermediate_states() {
    return intermediate_states;
  }

  std::optional<std::vector<observe_result>> get_final_expectation_values() { 
    return final_expectation_values;
  }

  std::optional<std::vector<std::vector<observe_result>>> get_expectation_values() {
    return expectation_values;
  }

};
} // namespace cudaq