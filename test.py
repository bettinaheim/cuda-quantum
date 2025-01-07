# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast, cudaq, textwrap, inspect, re, sys
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

def dynamic_kernel(dynops_basename, matrices):

    def decorator(func):

        dynamic_operations = ''
        for op_idx, matrix in enumerate(matrices):
            op_name = f"{dynops_basename}_{op_idx}"
            cudaq.register_operation(op_name, matrix)
            #lines = re.sub(op_name + r"(\(.+\))", f"{eval(op_name, globals, locals)}\g<1>", lines)
            # TODO: 
            # Either have register_operation return a kernel, or have another option 
            # to get a kernel based on the name. 
            dynamic_operations += f"{op_name},"

        dynops_definition = ast.parse(f"{dynops_basename} = [{dynamic_operations[:-1]}]").body[0]
        lines = inspect.getsource(func)
        lines = textwrap.dedent(lines)
        parsed_func = ast.parse(lines)
        parsed_func.body[0].body.insert(0, dynops_definition)
        lines = ast.unparse(parsed_func)
        print(lines)
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
        # Case 1: op is in globalKernelRegistry - fails on pop
        ops = [k1]
        # Case 2: op is in globalRegisteredOperations - add to visit_Name same as for globalKernelRegistry?
        ops = [custom_op]
        # Case 3: op is a built-in op - not sure what to do with that
        ops = [x]
        # Case 4: op is passed as arg or local variable (i.e. is in symbolTable?)
        ops = [ops[0]]
        ops[0](0.5)
        custom_op(q)

    print(cudaq.sample(k4))

    # Version 3:
    # use custom decorator instead of registration, locally create kernel that uses it
    custom_ops_matrices = [
        expm((1.j * angle * cudaq.spin.y(0)).to_matrix({0:2})) 
    ]

    @dynamic_kernel("custom_ops", custom_ops_matrices)
    def k3():
        q = cudaq.qubit()
        # FIXME: I think Marwa may want to write a string here to be able to loop over an array of dyn ops
        custom_ops[0](q)

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
    @dynamic_kernel("custom_ops", custom_ops_matrices)
    def dynamic_op(q: cudaq.qubit):
        custom_ops[0](q)

    # THIS HAS A BUG!
    # probably somewhere in the caching (my_exp_0 is applied despite the source code saying it should be my_exp_2)
    print(cudaq.sample(k2, dynamic_op))

