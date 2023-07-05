/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -v %s -o %basename_t.x --target quantinuum --emulate && ./%basename_t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

struct init_state {
  __qpu__ void operator()(cudaq::qreg<> &qubits, double theta) {
    ry(theta, qubits[0]);
    h<cudaq::ctrl>(qubits[0], qubits[1]);
    x(qubits[1]);

    ry(theta, qubits[2]);
    h<cudaq::ctrl>(qubits[2], qubits[3]);
    x(qubits[3]);

    ry(theta, qubits[4]);
    h<cudaq::ctrl>(qubits[4], qubits[5]);
    x(qubits[5]);

    ry(theta, qubits[6]);
    h<cudaq::ctrl>(qubits[6], qubits[7]);
    x(qubits[7]);
  }
};

__qpu__ void reflect_uniform(cudaq::qreg<> &qubits, double theta) {
  cudaq::adjoint(init_state{}, qubits, theta);
  x(qubits);
  z<cudaq::ctrl>(qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5], qubits[6], qubits[7]);
  x(qubits);
  init_state{}(qubits, theta);
}

bool oracle_classical(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
  bool c_01 = v0 != v1;
  bool c_123 = (v1 != v2) && (v1 != v3) && (v2 != v3);
  return c_01 && c_123;
}

__qpu__ void oracle(cudaq::qreg<> &cs, cudaq::qubit &target) {
  x<cudaq::ctrl>(cs[0], !cs[1], cs[2], !cs[3], cs[5], target);
  x<cudaq::ctrl>(cs[0], !cs[1], cs[2], !cs[3], cs[7], target);
  x<cudaq::ctrl>(cs[0], !cs[1], !cs[3], cs[4], cs[7], target);
  x<cudaq::ctrl>(cs[1], cs[2], cs[3], cs[4], target);
  x<cudaq::ctrl>(cs[1], !cs[2], cs[3], cs[6], target);
  x<cudaq::ctrl>(!cs[1], cs[2], cs[4], cs[7], target);
  x<cudaq::ctrl>(cs[0], cs[1], cs[2], cs[3], cs[5], target);
  x<cudaq::ctrl>(cs[1], !cs[2], cs[5], cs[6], target);
  x<cudaq::ctrl>(!cs[1], cs[3], cs[4], target);
  x<cudaq::ctrl>(cs[1], cs[4], cs[7], target);
  x<cudaq::ctrl>(cs[0], !cs[1], !cs[3], cs[5], cs[6], target);
  x<cudaq::ctrl>(!cs[2], cs[3], !cs[5], cs[6], target);
  x<cudaq::ctrl>(cs[0], cs[1], cs[2], cs[3], cs[7], target);
  x<cudaq::ctrl>(cs[0], cs[1], cs[3], cs[4], !cs[7], target);
  x<cudaq::ctrl>(cs[2], !cs[7], target);
  x<cudaq::ctrl>(cs[0], cs[1], cs[3], !cs[5], cs[6], target);
  x<cudaq::ctrl>(cs[2], !cs[3], cs[6], target);
  x<cudaq::ctrl>(cs[2], !cs[5], !cs[6], target);
  x<cudaq::ctrl>(!cs[2], cs[3], cs[4], cs[7], target);
}

__qpu__ void grover(double theta) {
  cudaq::qreg qubits(8);
  cudaq::qubit ancilla;

  // Initialization
  x(ancilla);
  h(ancilla);
  init_state{}(qubits, theta);

  // Iterations
  oracle(qubits, ancilla);
  reflect_uniform(qubits, theta);

  oracle(qubits, ancilla);
  reflect_uniform(qubits, theta);
  mz(qubits);
};

int main() {
  double theta = 2. * std::acos(1. / std::sqrt(3));
  auto result = cudaq::sample(1000, grover, theta);
  std::vector<std::string> strings;
  for (auto &&[bits, count] : result) {
    strings.push_back(bits);
  }
  std::sort(strings.begin(), strings.end(), [&](auto& a, auto& b) {
    return result.count(a) > result.count(b);
  });
  for (auto i = 0; i < 12; ++i) {
    for (auto j = 0; j < 8; j += 2)
      std::cout << strings[i].substr(j, 2) << " ";
    std::cout << '\n';
  }
  return 0;
}

// CHECK-DAG: 01 10 11 01
// CHECK-DAG: 10 11 01 10
// CHECK-DAG: 11 10 11 01
// CHECK-DAG: 01 11 01 10
// CHECK-DAG: 01 10 01 11
// CHECK-DAG: 01 11 10 01
// CHECK-DAG: 10 11 10 01
// CHECK-DAG: 11 01 11 10
// CHECK-DAG: 11 10 01 11
// CHECK-DAG: 10 01 10 11
// CHECK-DAG: 10 01 11 10
// CHECK-DAG: 11 01 10 11
