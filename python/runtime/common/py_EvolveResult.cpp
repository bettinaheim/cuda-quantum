/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <optional>
#include <pybind11/stl.h>
#include "py_EvolveResult.h"
#include "common/EvolveResult.h"
#include "cudaq/algorithms/evolve.h"

namespace py = pybind11;

namespace cudaq {
/// @brief Bind the `cudaq::evolve_result` and `cudaq::async_evolve_result`
/// data classes to python as `cudaq.EvolveResult` and `cudaq.AsyncEvolveResult`.
void bindEvolveResult(py::module &mod) {
  py::class_<evolve_result>(
      mod, "EvolveResult",
      "Stores the execution data from an invocation of :func:`evolve`.\n")
      /* 
        FIXME: doc comment;

        Instantiates an EvolveResult representing the output generated when evolving a single 
        initial state under a set of operators. See `cudaq.evolve` for more detail.

        Arguments:
            state: A single state or a sequence of states of a quantum system. If a single 
                state is given, it represents the final state of the system after time evolution.
                If a sequence of states are given, they represent the state of the system after
                each steps in the schedule specified in the call to `cudaq.evolve`.
            expectation: An one-dimensional array of results from the call to 
                `cudaq.observe` or a sequence of arrays thereof. If a single array 
                is provided, it contains the results from the calls to `cudaq.observe`
                at the end of the evolution for each observable defined in the call 
                to `cudaq.evolve`. If a sequence of arrays is provided, they represent 
                the results computed at each step in the schedule passed to 
                `cudaq.evolve`.
      */
      .def(py::init<state>())
      .def(py::init<std::vector<state>>())
      .def(py::init<state, std::vector<observe_result>>())
      .def(py::init<std::vector<state>, std::vector<std::vector<observe_result>>>())
      .def(
          "final_state", [](evolve_result &self) { return self.get_final_state(); },
          "Stores the final state produced by a call to :func:`evolve`. "
          "Represent the state of a quantum system after time evolution under "
          "a set of operators, see the :func:`evolve` documentation for more "
          "detail.\n")
      .def(
          "intermediate_states", [](evolve_result &self) { return self.get_intermediate_states(); },
          "Stores all intermediate states, meaning the state after each step "
          "in a defined schedule, produced by a call to :func:`evolve`, "
          "including the final state. This property is only populated if "
          "saving intermediate results was requested in the call to "
          ":func:`evolve`.\n")
      .def(
          "final_expectation_values", [](evolve_result &self) { return self.get_final_expectation_values(); },
          "Stores the final expectation values, that is the results produced by "
          "calls to :func:`observe`, triggered by a call to :func:`evolve`. Each "
          "entry corresponds to one observable provided in the :func:`evolve` "
          "call. This value will be None if no observables were specified in the "
          "call.\n")
      .def(
          "expectation_values", [](evolve_result &self) { return self.get_expectation_values(); },
          "Stores the expectation values, that is the results from the calls to "
          ":func:`observe`, at each step in the schedule produced by a call to "
          ":func:`evolve`, including the final expectation values. Each entry "
          "corresponds to one observable provided in the :func:`evolve` call. " 
          "This property is only populated saving intermediate results was "
          "requested in the call to :func:`evolve`. This value will be None "
          "if no intermediate results were requested, or if no observables "
          "were specified in the call.\n");

  py::class_<async_evolve_result>(
      mod, "AsyncEvolveResult",
      "Stores the execution data from an invocation of :func:`evolve_async`.\n")
      .def(
          "get", [](async_evolve_result &self) { return self.get(); },
          py::call_guard<py::gil_scoped_release>(),
          "Retrieve the evolution result from the asynchronous evolve execution\n.");
}

} // namespace cudaq
