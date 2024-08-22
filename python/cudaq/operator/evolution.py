from __future__ import annotations
import numpy, scipy, sys, uuid
from numpy.typing import NDArray
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .expressions import Operator
from .helpers import _OperatorHelpers, NumericType
from .schedule import Schedule
from ..kernel.register_op import register_operation
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
# FIXME: PyKernelDecorator is documented but not accessible in the API due to shadowing by the kernel module.
#from ..kernel.kernel_decorator import PyKernelDecorator
from ..kernel.kernel_builder import PyKernel, make_kernel
from ..runtime.observe import observe

def _compute_step_matrix(hamiltonian: Operator, dimensions: Mapping[int, int], parameters: Mapping[str, NumericType]) -> NDArray[complexfloating]:
    op_matrix = hamiltonian.to_matrix(dimensions, **parameters)
    # FIXME: Use approximative approach (series expansion, integrator), 
    # and maybe use GPU acceleration for matrix manipulations if it makes sense.
    return scipy.linalg.expm(-1j * op_matrix)

# FIXME: move to C++
def _evolution_kernel(num_qubits: int, 
                      compute_step_matrix: Callable[[Mapping[str, NumericType]], NDArray[numpy.complexfloating]], 
                      schedule: Iterable[Mapping[str, NumericType]],
                      split_into_steps = False) -> Generator[PyKernel]:
    kernel_base_name = "".join(filter(str.isalnum, str(uuid.uuid4())))
    def register_operations():
        for step_idx, parameters in enumerate(schedule):
            # We could make operators hashable and try to use that to do some kernel caching, 
            # but there is no guarantee that if the hash is equal, the operator is equal.
            # Overall it feels like a better choice to just take a uuid here.
            operation_name = f"evolve_{kernel_base_name}_{step_idx}"
            register_operation(operation_name, compute_step_matrix(parameters))
            yield operation_name
    operation_names = register_operations()

    evolution, initial_state = make_kernel(cudaq_runtime.State)
    qs = evolution.qalloc(initial_state)
    for operation_name in operation_names:
        # FIXME: #(*qs) causes infinite loop
        # FIXME: It would be nice if a registered operation could take a vector of qubits?
        targets = [qs[i] for i in range(num_qubits)]
        evolution.__getattr__(f"{operation_name}")(*targets)
        if split_into_steps:
            yield evolution
            evolution, initial_state = make_kernel(cudaq_runtime.State)
            qs = evolution.qalloc(initial_state)
    if not split_into_steps: yield evolution


# Top level API for the CUDA-Q master equation solver.
def evolve(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.States],
           collapse_operators: Sequence[Operator] = [],
           observables: Sequence[Operator] = [], 
           store_intermediate_results = False) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult]:
    """
    Computes the time evolution of one or more initial state(s) under the defined 
    operator(s). 

    Arguments:
        hamiltonian: Operator that describes the behavior of a quantum system
            without noise.
        dimensions: A mapping that specifies the number of levels, that is
            the dimension, of each degree of freedom that any of the operator 
            arguments acts on.
        schedule: A sequence that generates a mapping of keyword arguments 
            to their respective value. The keyword arguments are the parameters
            needed to evaluate any of the operators passed to `evolve`.
            All required parameters for evaluating an operator and their
            documentation, if available, can be queried by accessing the
            `parameter` property of the operator.
        initial_state: A single state or a sequence of states of a quantum system.
        collapse_operators: A sequence of operators that describe the influence of 
            noise on the quantum system.
        observables: A sequence of operators for which to compute their expectation
            value during evolution. If `store_intermediate_results` is set to True,
            the expectation values are computed after each step in the schedule, 
            and otherwise only the final expectation values at the end of the 
            evolution are computed.

    Returns:
        A single evolution result if a single initial state is provided, or a sequence
        of evolution results representing the data computed during the evolution of each
        initial state. See `EvolveResult` for more information about the data computed
        during evolution.
    """
    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError("time evolution is currently only supported on simulator targets")
    elif simulator == "nvidia-dynamics": # FIXME: update here and below once we know the target name
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using cuSuperoperator for the execution, 
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: tensornet can potentially handle qudits
        raise ValueError("computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems")
    # Unless we are using cuSuperoperator for the execution, 
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0:
        raise ValueError("collapse operators can only be defined when using the nvidia-dynamics target")

    num_qubits = len(hamiltonian.degrees)
    parameters = [mapping for mapping in schedule]
    observable_spinops = [lambda step_parameters: op._to_spinop(dimensions, **step_parameters) for op in observables]
    compute_step_matrix = lambda step_parameters: _compute_step_matrix(hamiltonian, dimensions, step_parameters)

    # FIXME: deal with a sequence of initial states
    if store_intermediate_results:
        evolution = _evolution_kernel(num_qubits, compute_step_matrix, parameters, split_into_steps = True)
        kernels = [kernel for kernel in evolution]
        if len(observables) == 0: return cudaq_runtime.evolve(initial_state, kernels)
        return cudaq_runtime.evolve(initial_state, kernels, parameters, observable_spinops)
    else:
        kernel = next(_evolution_kernel(num_qubits, compute_step_matrix, parameters))
        if len(observables) == 0: return cudaq_runtime.evolve(initial_state, kernel)
        # FIXME: permit to compute expectation values for operators defined as matrix
        return cudaq_runtime.evolve(initial_state, kernel, parameters[-1], observable_spinops)

def evolve_async(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.State],
           collapse_operators: Sequence[Operator] = [],
           observables: Sequence[Operator] = [], 
           store_intermediate_results = False) -> cudaq_runtime.AsyncEvolveResult | Sequence[cudaq_runtime.AsyncEvolveResult]:
    """
    Asynchronously computes the time evolution of one or more initial state(s) 
    under the defined operator(s). See `cudaq.evolve` for more details about the
    parameters passed here.
    
    Returns:
        The handle to a single evolution result if a single initial state is provided, 
        or a sequence of handles to the evolution results representing the data computed 
        during the evolution of each initial state. See the `EvolveResult` for more 
        information about the data computed during evolution.
    """
    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError("time evolution is currently only supported on simulator targets")
    elif simulator == "nvidia-dynamics": # FIXME: update once we know the target name
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using cuSuperoperator for the execution, 
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: tensornet can potentially handle qudits
        raise ValueError("computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems")
    # Unless we are using cuSuperoperator for the execution, 
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0:
        raise ValueError("collapse operators can only be defined when using the nvidia-dynamics target")

    num_qubits = len(hamiltonian.degrees)
    parameters = [mapping for mapping in schedule]
    observable_spinops = [lambda step_parameters: op._to_spinop(dimensions, **step_parameters) for op in observables]
    compute_step_matrix = lambda step_parameters: _compute_step_matrix(hamiltonian, dimensions, step_parameters)

    # FIXME: deal with a sequence of initial states
    if store_intermediate_results:
        evolution = _evolution_kernel(num_qubits, compute_step_matrix, parameters, split_into_steps = True)
        kernels = [kernel for kernel in evolution]
        if len(observables) == 0: return cudaq_runtime.evolve_async(initial_state, kernels)
        return cudaq_runtime.evolve_async(initial_state, kernels, parameters, observable_spinops)
    else:
        kernel = next(_evolution_kernel(num_qubits, compute_step_matrix, parameters))
        if len(observables) == 0: return cudaq_runtime.evolve_async(initial_state, kernel)
        # FIXME: permit to compute expectation values for operators defined as matrix
        return cudaq_runtime.evolve_async(initial_state, kernel, parameters[-1], observable_spinops)
