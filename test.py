# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, textwrap, inspect, re, sys
from scipy.linalg import expm
from typing import Callable

angles= [0.5, 1., 1.5]
dyn_ops_id = 0


@cudaq.kernel
def k1(angle: float):
    qs = cudaq.qvector(1)
    exp_pauli(angle, qs, "Y")

@cudaq.kernel
def k2(custom_op: Callable[[cudaq.qubit], None]):
    q = cudaq.qubit()
    custom_op(q)

def dynamic_kernel(dyn_ops):

    def decorator(func):

        lines = inspect.getsource(func)
        lines = textwrap.dedent(lines)
        for op_name in dyn_ops:
            cudaq.register_operation(op_name, dyn_ops[op_name])
            #lines = re.sub(op_name + r"(\(.+\))", f"{eval(op_name, globals, locals)}\g<1>", lines)

        return cudaq.PyKernelDecorator("dynamic_kernel",
                                kernelName = func.__name__,
                                funcSrc = lines,
                                signature = {},
                                location = (__file__, sys._getframe().f_lineno))
    return decorator

for i, angle in enumerate(angles):

    # Version 1: 
    # use built-in pauli exponential
    print(cudaq.sample(k1, angle))

    # Version 2:
    # standard way of defining exponential as a custom op
    cudaq.register_operation("custom_op", expm((1.j * angle * cudaq.spin.y(0)).to_matrix({0:2})))

    @cudaq.kernel
    def k4():
        q = cudaq.qubit()
        custom_op(q)

    print(cudaq.sample(k4))

    # Version 3:
    # use custom decorator instead of registration, locally create kernel that uses it
    dynamic_ops = {
        "dyn_op": expm((1.j * angle * cudaq.spin.y(0)).to_matrix({0:2})) 
    }

    @dynamic_kernel(dynamic_ops)
    def k3():
        q = cudaq.qubit()
        # FIXME: I think Marwa may want to write a string here to be able to loop over an array of dyn ops
        dyn_op(q)

    print(cudaq.sample(k3))

    # Version 4: 
    # register custom op, pass to global kernel that uses it
    @cudaq.kernel
    def dynamic_op(q: cudaq.qubit):
        custom_op(q)

    # This fails to jit compile. Not sure why it should - may be a bug.
    # Maybe this is what you wanted (besides a more general version 1) and ran into trouble with?
    # print(cudaq.sample(k2, dynamic_op))

    # Version 5: 
    # use custom decorator, pass custom op to global kernel that uses it
    @dynamic_kernel(dynamic_ops)
    def dynamic_op(q: cudaq.qubit):
        dyn_op(q)

    # THIS HAS A BUG!
    # probably somewhere in the caching (my_exp_0 is applied despite the source code saying it should be my_exp_2)
    print(cudaq.sample(k2, dynamic_op))

