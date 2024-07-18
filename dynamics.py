import numpy, itertools, types, functools, copy
from typing import Callable
from numpy.typing import NDArray

# FIXME: 
# This BuildIns class is just a stand-in for now - ignore this class; 
# it will be heavily revised in future iterations.
class BuiltIns:
    ops = {
        ("spin_x", "2"): lambda _: numpy.array([[0,1],[1,0]]),
        ("spin_y", "2"): lambda _: numpy.array([[0,1j],[-1j,0]]),
        ("spin_z", "2"): lambda _: numpy.array([[1,0],[0,-1]]),
        ("spin_i", "2"): lambda _: numpy.array([[1,0],[0,1]]),
    }

class ScalarOperator():
    pass

class ElementaryOperator():
    pass

class ProductOperator():
    pass

class OperatorSum():
    pass

class OperatorSum:
    _terms: list[ProductOperator]

    def __init__(self, terms: list[ProductOperator]):
        if len(terms) == 0:
            raise ValueError("Need at least one term.")
        self._terms = terms

    def concretize(self, time: complex, levels: dict[int, int]) -> NDArray[complex]:
        degrees = set([degree for term in self._terms for op in term._operators for degree in op._degrees])
        padded_terms = [] # We need to make sure all matrices are of the same size to sum them up.
        for term in self._terms:
            for degree in degrees:
                if not degree in [degree for op in term._operators for degree in op._degrees]:
                    term *= spin.i(degree)
            padded_terms.append(term)
        return sum([term.concretize(time, levels) for term in padded_terms])

    def __mul__(self, other: OperatorSum):
        if type(other) == ScalarOperator:
            return self * OperatorSum([ProductOperator([other])])
        elif type(other) == ElementaryOperator:
            return self * OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self * OperatorSum([other])
        return OperatorSum([self_term * other_term for self_term in self._terms for other_term in other._terms])
    
    def __add__(self, other: OperatorSum):
        if type(other) == ScalarOperator:
            return self + OperatorSum([ProductOperator([other])])
        elif type(other) == ElementaryOperator:
            return self + OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self + OperatorSum([other])
        return OperatorSum(self._terms + other._terms)

class ProductOperator(OperatorSum):
    _operators: list[ElementaryOperator]

    def __init__(self, operators : list[ElementaryOperator]):
        if len(operators) == 0:
            raise ValueError("Need at least one operator.")
        self._operators = operators
        super().__init__([self])

    def concretize(self, time: complex, levels: dict[int, int]) -> NDArray[complex]:
        def generate_all_states(degrees: list[int]):
            if len(degrees) == 0:
                return []
            states = [[str(state)] for state in range(levels[degrees[0]])]
            for d in degrees[1:]:
                prod = itertools.product(states, [str(state) for state in range(levels[d])])
                states = [current + [new] for current, new in prod]
            return [''.join(state) for state in states]

        def padded_matrix(op: ElementaryOperator, degrees: list[int]):
            op_matrix = op.concretize(time, levels)
            op_degrees = op._degrees.copy() # Determines the initial qubit ordering of op_matrix.
            for degree in degrees:
                if not degree in [d for d in op._degrees]:
                    op_matrix = numpy.kron(op_matrix, spin.i(degree).concretize(time, levels))
                    op_degrees.append(degree)
            # Need to permute the matrix such that the qubit ordering of all matrices is the same.
            if op_degrees != degrees:
                # I'm sure there is a more efficient way, but needed something correct first.
                states = generate_all_states(degrees) # ordered according to degrees
                indices = dict([(d, idx) for idx, d in enumerate(degrees)])
                reordering = [indices[op_degree] for op_degree in op_degrees]
                # [degrees[i] for i in reordering] produces op_degrees
                op_states = [''.join([state[i] for i in reordering]) for state in states]
                state_indices = dict([(state, idx) for idx, state in enumerate(states)])
                permutation = [state_indices[op_state] for op_state in op_states]
                # [states[i] for i in permutation] produces op_states
                for i in range(numpy.size(op_matrix, 1)):
                    op_matrix[:,i] = op_matrix[permutation,i]
                for i in range(numpy.size(op_matrix, 0)):
                    op_matrix[i,:] = op_matrix[i,permutation]
            return op_matrix
        
        degrees = list(set([degree for op in self._operators for degree in op._degrees]))
        degrees.sort() # This sorting determines the qubit ordering of the final matrix.
        # FIXME: make the overall degrees of the product op accessible?
        # FIXME: check endianness ...
        matrix = padded_matrix(self._operators[0], degrees)
        for op in self._operators[1:]:
            matrix = numpy.dot(matrix, padded_matrix(op, degrees))
        return matrix

    def __mul__(self, other: ProductOperator):
        if type(other) == ScalarOperator:
            return self * ProductOperator([other])
        elif type(other) == ElementaryOperator:
            return self * ProductOperator([other])
        elif type(other) != ProductOperator:
            return OperatorSum([self]) * other
        return ProductOperator(self._operators + other._operators)

    def __add__(self, other: ProductOperator):
        if type(other) == ScalarOperator:
            return self + ProductOperator([other])
        elif type(other) == ElementaryOperator:
            return self + OperatorSum([other])
        elif type(other) != ProductOperator:
            return OperatorSum([self]) + other
        return OperatorSum([self, other])

class ElementaryOperator(ProductOperator):
    _degrees: list[int]
    _builtin_id: str

    def __init__(self, builtin_id: str, degrees: list[int]):
        self._builtin_id = builtin_id
        self._degrees = degrees
        self._degrees.sort() # sorting so that we have a unique ordering for builtin
        super().__init__([self])

    def concretize(self, time: complex, levels: dict[int, int]) -> NDArray[complex]:
        missing_degrees = [degree not in levels for degree in self._degrees]
        if any(missing_degrees):
            raise ValueError(f'Missing levels for degree(s) {[self._degrees[i] for i, x in enumerate(missing_degrees) if x]}')

        relevant_levels = [levels[degree] for degree in self._degrees]
        ops_key = (self._builtin_id, "".join([str(l) for l in relevant_levels]))
        if not ops_key in BuiltIns.ops:
           raise ValueError(f'No built-in operator {self._builtin_id} for {len(relevant_levels)} degrees of freedom with levels {relevant_levels}.')
        return BuiltIns.ops[ops_key](time)

    def __mul__(self, other: ElementaryOperator):
        if type(other) == ScalarOperator:
            return ProductOperator([self]) * ProductOperator([other])
        elif type(other) != ElementaryOperator:
            return ProductOperator([self]) * other
        return ProductOperator([self, other])

    def __add__(self, other: ElementaryOperator):
        if type(other) == ScalarOperator:
            return ProductOperator([self]) + ProductOperator([other])
        elif type(other) != ElementaryOperator:
            return ProductOperator([self]) + other
        op1 = ProductOperator([self])
        op2 = ProductOperator([other])
        return OperatorSum([op1, op2])

class ScalarOperator(ProductOperator):
    _degrees: list[int] # Always empty; here for consistency with other operators.
    _generator: Callable[[complex], complex]

    def __init__(self, generator: Callable[[complex], complex]):
        self._degrees = []
        self._generator = generator
        super().__init__([self])

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, generator: Callable[[complex], complex]):
        self._generator = generator

    # The argument `levels` here is only passed for consistency with parent classes.
    def concretize(self, time: complex, levels: dict[int, int] = None):
        return self._generator(time)

    def __matmul__(self, other: ScalarOperator) -> ScalarOperator:
        if type(other) != ScalarOperator:
            raise ValueError("Right hand side of composition must be a ScalarOperator.")
        generator = lambda time: self._generator(other._generator(time))
        return ScalarOperator(generator)

    def __mul__(self, other: ScalarOperator) -> ScalarOperator:
        if type(other) != ScalarOperator:
            return ProductOperator([self]) * other
        generator = lambda time: self._generator(time) * other._generator(time)
        return ScalarOperator(generator)
    
    def __add__(self, other: ScalarOperator) -> ScalarOperator:
        if type(other) != ScalarOperator:
            return ProductOperator([self]) + other
        generator = lambda time: self._generator(time) + other._generator(time)
        return ScalarOperator(generator)

# Operators as defined here: 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html

class spin:

    @classmethod
    def x(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("spin_x", [degree])
    @classmethod
    def y(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("spin_y", [degree])
    @classmethod
    def z(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("spin_z", [degree])
    @classmethod
    def i(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("spin_i", [degree])
    
levels = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
time = 1.0

print(f'spinX(1): {spin.x(1).concretize(time, levels)}')
print(f'spinY(2): {spin.y(2).concretize(time, levels)}')

print(f'spinZ(0) * spinZ(0): {(spin.z(0) * spin.z(0)).concretize(time, levels)}')
print(f'spinZ(0) * spinZ(1): {(spin.z(0) * spin.z(1)).concretize(time, levels)}')
print(f'spinZ(0) * spinY(1): {(spin.z(0) * spin.y(1)).concretize(time, levels)}')

op1 = ProductOperator([spin.x(0), spin.i(1)])
op2 = ProductOperator([spin.i(0), spin.x(1)])
print(f'spinX(0) + spinX(1): {op1.concretize(time, levels) + op2.concretize(time, levels)}')
op3 = ProductOperator([spin.x(1), spin.i(0)])
op4 = ProductOperator([spin.i(1), spin.x(0),])
print(f'spinX(1) + spinX(0): {op1.concretize(time, levels) + op2.concretize(time, levels)}')

print(f'spinX(0) + spinX(1): {(spin.x(0) + spin.x(1)).concretize(time, levels)}')
print(f'spinX(0) * spinX(1): {(spin.x(0) * spin.x(1)).concretize(time, levels)}')
print(f'spinX(0) * spinI(1) * spinI(0) * spinX(1): {(op1 * op2).concretize(time, levels)}')

print(f'spinX(0) * spinI(1): {op1.concretize(time, levels)}')
print(f'spinI(0) * spinX(1): {op2.concretize(time, levels)}')
print(f'spinX(0) * spinI(1) + spinI(0) * spinX(1): {(op1 + op2).concretize(time, levels)}')

op5 = spin.x(0) * spin.x(1)
op6 = spin.z(0) * spin.z(1)
print(f'spinX(0) * spinX(1): {op5.concretize(time, levels)}')
print(f'spinZ(0) * spinZ(1): {op6.concretize(time, levels)}')
print(f'spinX(0) * spinX(1) + spinZ(0) * spinZ(1): {(op5 + op6).concretize(time, levels)}')

op7 = spin.x(0) + spin.x(1)
op8 = spin.z(0) + spin.z(1)
print(f'spinX(0) + spinX(1): {op7.concretize(time, levels)}')
print(f'spinZ(0) + spinZ(1): {op8.concretize(time, levels)}')
print(f'spinX(0) + spinX(1) + spinZ(0) + spinZ(1): {(op7 + op8).concretize(time, levels)}')
print(f'(spinX(0) + spinX(1)) * (spinZ(0) + spinZ(1)): {(op7 * op8).concretize(time, levels)}')

print(f'spinX(0) * (spinZ(0) + spinZ(1)): {(spin.x(0) * op8).concretize(time, levels)}')
print(f'(spinZ(0) + spinZ(1)) * spinX(0): {(op8 * spin.x(0)).concretize(time, levels)}')

op9 = spin.z(1) + spin.z(2)
print(f'(spinX(0) + spinX(1)) * spinI(2): {numpy.kron(op7.concretize(time, levels), spin.i(2).concretize(time, levels))}')
print(f'(spinX(0) + spinX(1)) * spinI(2): {(op7 * spin.i(2)).concretize(time, levels)}')
print(f'(spinX(0) + spinX(1)) * spinI(2): {(spin.i(2) * op7).concretize(time, levels)}')
print(f'spinI(0) * (spinZ(1) + spinZ(2)): {numpy.kron(spin.i(0).concretize(time, levels), op9.concretize(time, levels))}')
print(f'(spinX(0) + spinX(1)) * (spinZ(1) + spinZ(2)): {(op7 * op9).concretize(time, levels)}')

so1 = ScalarOperator(lambda t: t)
print(f'Scalar op (t -> t)({time}): {so1.concretize(time)}')
print(f'Trivial prod op (t -> t)({time}): {(ProductOperator([so1])).concretize(time, {})}')
print(f'Trivial prod op (t -> t)({time + 1}): {(ProductOperator([so1])).concretize(time + 1, {})}')
print(f'(t -> t)(1j) * spinX(0): {(so1 * spin.x(0)).concretize(1j, levels)}')
print(f'spinX(0) * (t -> t)(1j): {(spin.x(0) * so1).concretize(1j, levels)}')
print(f'spinX(0) + (t -> t)(1j): {(spin.x(0) + so1).concretize(1j, levels)}')
print(f'(t -> t)(1j) + spinX(0): {(so1 + spin.x(0)).concretize(1j, levels)}')
print(f'spinX(0) + (t -> t)(1j): {(spin.x(0) + so1).concretize(1j, levels)}')
print(f'(t -> t)(1j) + spinX(0): {(so1 + spin.x(0)).concretize(1j, levels)}')
print(f'(t -> t)({2*time}) * (spinX(0) + spinX(1)) * (spinZ(1) + spinZ(2)): {(so1 * op7 * op9).concretize(2*time, levels)}')
print(f'(spinX(0) + spinX(1)) * (t -> t)({2*time}) * (spinZ(1) + spinZ(2)): {(op7 * so1 * op9).concretize(2*time, levels)}')
print(f'(spinX(0) + spinX(1)) * (spinZ(1) + spinZ(2)) * (t -> t)({2*time}): {(op7 * op9 * so1).concretize(2*time, levels)}')

op10 = so1 * spin.x(0)
so1.generator = lambda t: 1./t
print(f'(t -> 1/t)(2) * spinX(0): {op10.concretize(2., levels)}')
so1_gen2 = so1.generator
so1.generator = lambda t: so1_gen2(2*t)
print(f'(t -> 1/(2t))(2) * spinX(0): {op10.concretize(2., levels)}')
so1.generator = lambda t: so1_gen2(t)
print(f'(t -> 1/t)(2) * spinX(0): {op10.concretize(2., levels)}')

so2 = ScalarOperator(lambda t: t**2)
op11 = spin.z(1) * so2
print(f'spinZ(0) * (t -> t^2)(2): {op11.concretize(2., levels)}')
op12 = so1 @ so2 * spin.z(1)
print(f'(t -> 1/(t^2))(2) * spinZ(0): {op12.concretize(2., levels)}')
so1.generator = lambda t: t
print(f'(t -> t^2)(2) * spinZ(0): {op12.concretize(2., levels)}')
so2.generator = so1.generator
print(f'(t -> t)(2) * spinZ(0): {op12.concretize(2., levels)}')


