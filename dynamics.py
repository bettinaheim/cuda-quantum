from __future__ import annotations
import inspect, itertools, numpy, os, operator, re, scipy # type: ignore
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Optional, SupportsComplex, Tuple, TypeVar
from numpy.typing import NDArray


class _OperatorHelpers:

    @staticmethod
    def aggregate_parameters(dicts: Iterable[dict[str, str]]):
        """
        Helper function used by all operator classes to return a dictionary with the
        used parameters and their respective description as defined in a doc comment.
        """
        param_descriptions : dict[str,str] = {}
        for descriptions in dicts:
            for key in descriptions:
                existing_desc = param_descriptions.get(key) or ""
                new_desc = descriptions[key]
                has_existing_desc = existing_desc is not None and existing_desc != ""
                has_new_desc = new_desc != ""
                if has_existing_desc and has_new_desc:
                    param_descriptions[key] = existing_desc + f'{os.linesep}---{os.linesep}' + new_desc
                elif has_existing_desc:
                    param_descriptions[key] = existing_desc
                else: 
                    param_descriptions[key] = new_desc
        return param_descriptions
    
    @staticmethod
    def parameter_docs(param_name: str, docs: Optional[str]) -> str:
        """
        Given the function documentation, tries to find the documentation for a 
        specific parameter. Expects Google docs style to be used.

        Returns:
            The documentation or an empty string if it failed to find it.
        """
        if param_name is None or docs is None:
            return ""
        
        # Using re.compile on the pattern before passing it to search
        # seems to behave slightly differently than passing the string.
        # Also, Python caches already used patterns, so compiling on
        # the fly seems fine.
        def keyword_pattern(word):
            return r"(?:^\s*" + word + ":\s*\r?\n)"
        def param_pattern(param_name): 
            return r"(?:^\s*" + param_name + r"\s*(\(.*\))?:)\s*(.*)$"

        param_docs = ""
        try: # Make sure failing to retrieve docs never cases an error.
            split = re.split(keyword_pattern("Arguments"), docs, flags=re.MULTILINE)
            if len(split) == 1:
                split = re.split(keyword_pattern("Args"), docs, flags=re.MULTILINE)
            if len(split) > 1:
                match = re.search(param_pattern(param_name), split[1], re.MULTILINE)
                if match is not None:
                    param_docs = match.group(2) + split[1][match.end(2):]
                    match = re.search(param_pattern("\S*?"), param_docs, re.MULTILINE)
                    if match is not None:
                        param_docs = param_docs[:match.start(0)]
                    param_docs = re.sub(r'\s+', ' ', param_docs)
        except: pass
        return param_docs.strip()

    @staticmethod
    def args_from_kwargs(fct, **kwargs):
        """
        Extracts the positional argument and keyword only arguments 
        for the given function from the passed kwargs. 
        """
        arg_spec = inspect.getfullargspec(fct)
        signature = inspect.signature(fct)
        if arg_spec.varargs is not None:
            raise ValueError("cannot extract arguments for a function with a *args argument")
        consumes_kwargs = arg_spec.varkw is not None
        if consumes_kwargs:
            kwargs = kwargs.copy() # We will modify and return a copy

        def find_in_kwargs(arg_name: str):
            # Try to get the argument from the kwargs passed during operator 
            # evaluation.
            arg_value = kwargs.get(arg_name)
            if arg_value is None:
                # If no suitable keyword argument was defined, check if the 
                # generator defines a default value for this argument.
                default_value = signature.parameters[arg_name].default
                if default_value is not inspect.Parameter.empty:
                    arg_value = default_value
            elif consumes_kwargs:
                del kwargs[arg_name]
            if arg_value is None:
                raise ValueError(f'missing keyword argument {arg_name}')
            return arg_value

        extracted_args = []
        for arg_name in arg_spec.args:
            extracted_args.append(find_in_kwargs(arg_name))
        if consumes_kwargs:
            return extracted_args, kwargs
        elif len(arg_spec.kwonlyargs) > 0:
            # If we can't pass all remaining kwargs, 
            # we need to create a separate dictionary for kwonlyargs.
            kwonlyargs = {}
            for arg_name in arg_spec.kwonlyargs:
                kwonlyargs[arg_name] = find_in_kwargs(arg_name)
            return extracted_args, kwonlyargs
        return extracted_args, {}

    @staticmethod
    def generate_all_states(degrees: list[int], dimensions: dict[int, int]):
        """
        Generates all possible states for the given dimensions ordered according to 
        the list of degrees (ordering is relevant if dimensions differ).
        """
        if len(degrees) == 0:
            return []
        states = [[str(state)] for state in range(dimensions[degrees[0]])]
        for d in degrees[1:]:
            prod = itertools.product(states, [str(state) for state in range(dimensions[d])])
            states = [current + [new] for current, new in prod]
        return [''.join(state) for state in states]

    @staticmethod
    def permute_matrix(matrix: NDArray[numpy.complexfloating], permutation: list[int]) -> NDArray[numpy.complexfloating]:
        """
        Permutes the given matrix according to the given permutation.
        If states is the current order of vector entries on which the given matrix
        acts, and permuted_states is the desired order of an array on which the
        permuted matrix should act, then the permutation is defined as the list for
        which [states[i] for i in permutation] produces permuted_states.
        """
        for i in range(numpy.size(matrix, 1)):
            matrix[:,i] = matrix[permutation,i]
        for i in range(numpy.size(matrix, 0)):
            matrix[i,:] = matrix[i,permutation]
        return matrix


TEval = TypeVar('TEval')

class OperatorArithmetics(ABC, Generic[TEval]):
    """
    This class serves as a monad base class for computing arbitrary values
    during operator evaluation.
    """

    @abstractmethod
    def evaluate(self: OperatorArithmetics[TEval], op: ElementaryOperator | ScalarOperator) -> TEval:
        """
        Accesses the relevant data to evaluate an operator expression in the leaf 
        nodes, that is in elementary and scalar operators.
        """
        pass

    @abstractmethod
    def add(self: OperatorArithmetics[TEval], val1: TEval, val2: TEval) -> TEval: 
        """
        Adds two operators that act on the same degrees of freedom.
        """
        pass

    @abstractmethod
    def mul(self: OperatorArithmetics[TEval], val1: TEval, val2: TEval) -> TEval: 
        """
        Multiplies two operators that act on the same degrees of freedom.
        """
        pass

    @abstractmethod
    def tensor(self: OperatorArithmetics[TEval], val1: TEval, val2: TEval) -> TEval: 
        """
        Computes the tensor product of two operators that act on different 
        degrees of freedom.
        """
        pass

class MatrixArithmetics(OperatorArithmetics['MatrixArithmetics.Evaluated']):
    """
    Encapsulates the functions needed to compute the matrix representation
    of an operator expression.
    """

    class Evaluated:
        """
        Stores the relevant data to compute the matrix representation of an
        operator expression.
        """

        def __init__(self: MatrixArithmetics.Evaluated, degrees: list[int], matrix: NDArray[numpy.complexfloating]) -> None:
            """
            Instantiates an object that contains the matrix representation of an
            operator acting on the given degrees of freedom.

            Arguments:
                degrees: The degrees of freedom that the matrix applies to.
                matrix: The matrix representation of an evaluated operator.
            """
            self._degrees = degrees
            self._matrix = matrix

        @property
        def degrees(self: MatrixArithmetics.Evaluated) -> list[int]: 
            """
            The degrees of freedom that the matrix of the Evaluated value applies to.
            """
            return self._degrees

        @property
        def matrix(self: MatrixArithmetics.Evaluated) -> NDArray[numpy.complexfloating]: 
            """
            The matrix representation of an evaluated operator, ordered according
            to the list of degrees of freedom associated with the Evaluated value.
            """
            return self._matrix

    def _canonicalize(self: MatrixArithmetics, op_matrix: NDArray[numpy.complexfloating], op_degrees: list[int]) -> Tuple[NDArray[numpy.complexfloating], list[int]]:
        """
        Given a matrix representation that acts on the given degrees or freedom, 
        sorts the degrees and permutes the matrix to match that canonical order.

        Returns:
            A tuple consisting of the permuted matrix as well as the list of degrees
            of freedom in canonical order.
        """
        # FIXME: check endianness ... (in the sorting/states here, and in the matrix definitions)
        canon_degrees = sorted(op_degrees)
        if op_degrees != canon_degrees:
            # There may be a more efficient way, but I needed something correct first.
            states = _OperatorHelpers.generate_all_states(canon_degrees, self._dimensions)
            indices = dict([(d, idx) for idx, d in enumerate(canon_degrees)])
            reordering = [indices[op_degree] for op_degree in op_degrees]
            # [degrees[i] for i in reordering] produces op_degrees
            op_states = [''.join([state[i] for i in reordering]) for state in states]
            state_indices = dict([(state, idx) for idx, state in enumerate(states)])
            permutation = [state_indices[op_state] for op_state in op_states]
            # [states[i] for i in permutation] produces op_states
            return _OperatorHelpers.permute_matrix(op_matrix, permutation), canon_degrees
        return op_matrix, canon_degrees

    def tensor(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated, op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Computes the tensor product of two evaluate operators that act on different 
        degrees of freedom using `numpy.kron`.
        """
        assert len(set(op1.degrees).intersection(op2.degrees)) == 0, \
            "Operators should not have common degrees of freedom."
        op_degrees = op1.degrees + op2.degrees
        op_matrix = numpy.kron(op1.matrix, op2.matrix)
        new_matrix, new_degrees = self._canonicalize(op_matrix, op_degrees)
        return MatrixArithmetics.Evaluated(new_degrees, new_matrix)

    def mul(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated, op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Multiplies two evaluated operators that act on the same degrees of freedom
        using `numpy.dot`.
        """
        # Elementary operators have sorted degrees such that we have a unique convention 
        # for how to define the matrix. Tensor products permute the computed matrix if 
        # necessary to guarantee that all operators always have sorted degrees.
        assert op1.degrees == op2.degrees, "Operators should have the same order of degrees."
        return MatrixArithmetics.Evaluated(op1.degrees, numpy.dot(op1.matrix, op2.matrix))

    def add(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated, op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Multiplies two evaluated operators that act on the same degrees of freedom
        using `numpy`'s array addition.
        """
        # Elementary operators have sorted degrees such that we have a unique convention 
        # for how to define the matrix. Tensor products permute the computed matrix if 
        # necessary to guarantee that all operators always have sorted degrees.
        assert op1.degrees == op2.degrees, "Operators should have the same order of degrees."
        return MatrixArithmetics.Evaluated(op1.degrees, op1.matrix + op2.matrix)

    def evaluate(self: MatrixArithmetics, op: ElementaryOperator | ScalarOperator) -> MatrixArithmetics.Evaluated: 
        """
        Computes the matrix of an ElementaryOperator or ScalarOperator using its 
        `to_matrix` method.
        """
        matrix = op.to_matrix(self._dimensions, **self._kwargs)
        return MatrixArithmetics.Evaluated(op._degrees, matrix)

    def __init__(self: MatrixArithmetics, dimensions: dict[int, int], **kwargs: SupportsComplex) -> None:
        """
        Instantiates a MatrixArithmetics instance that can act on the given
        dimensions.

        Arguments:
            dimensions: A dictionary that specifies the number of levels, that 
                is the dimension, of each degree of freedom that the evaluated 
                operator can act on.
            **kwargs: Keyword arguments needed to evaluate, that is access data in,
                the leaf nodes of the operator expression. Leaf nodes are 
                values of type ElementaryOperator or ScalarOperator.
        """
        self._dimensions = dimensions
        self._kwargs = kwargs

class PrettyPrint(OperatorArithmetics[str]):

    def tensor(self, op1: str, op2: str) -> str:
        def add_parens(str_value: str):
            outer_str = re.sub(r'\(.+?\)', '', str_value)
            if " + " in outer_str or " * " in outer_str: return f"({str_value})"
            else: return str_value
        return f"{add_parens(op1)} x {add_parens(op2)}"

    def mul(self, op1: str, op2: str) -> str:
        def add_parens(str_value: str):
            outer_str = re.sub(r'\(.+?\)', '', str_value)
            if " + " in outer_str or " x " in outer_str: return f"({str_value})"
            else: return str_value
        return f"{add_parens(op1)} * {add_parens(op2)}"

    def add(self, op1: str, op2: str) -> str:
        return f"{op1} + {op2}"

    def evaluate(self, op: ElementaryOperator | ScalarOperator): return str(op)


class OperatorSum:
    """
    Represents an operator expression consisting of a sum of terms, 
    where each term is a product of elementary and scalar operators. 
    """

    def __init__(self: OperatorSum, terms: list[ProductOperator] = []) -> None:
        """
        Instantiates an operator expression that represents a sum of terms,
        where each term is a product of elementary and scalar operators. 

        Arguments:
            terms: The ProductOperators that should be summed up when 
                evaluating the operator expression.
        """
        if len(terms) == 0:
            terms = [ProductOperator([operators.const(0)])]
        self._terms = terms

    def __eq__(self: OperatorSum, other: Any) -> bool:
        """
        Returns:
            True, if the other value is an OperatorSum with the same terms, 
            and False otherwise.
        """
        return type(other) == OperatorSum and self._terms == other._terms

    @property
    def parameters(self: OperatorSum) -> dict[str, str]:
        """
        A dictionary that contains the documentation comment for each parameter 
        needed to evaluate the operator expression.
        """
        return _OperatorHelpers.aggregate_parameters([term.parameters for term in self._terms])

    def _evaluate(self: OperatorSum, arithmetics : OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function used for evaluating operator expressions and computing arbitrary values
        during evaluation. The value to be computed is defined by the OperatorArithmetics.
        The evaluation guarantees that addition and multiplication of two operators will only
        be called when both operators act on the same degrees of freedom, and the tensor product
        will only be computed if they act on different degrees of freedom. 
        """
        degrees = set([degree for term in self._terms for op in term._operators for degree in op._degrees])
        padded_terms = [] # We need to make sure all matrices are of the same size to sum them up.
        for term in self._terms:
            for degree in degrees:
                if not degree in [op_degree for op in term._operators for op_degree in op._degrees]:
                    term *= operators.identity(degree)
            padded_terms.append(term)
        sum = padded_terms[0]._evaluate(arithmetics)
        for term in padded_terms[1:]:
            sum = arithmetics.add(sum, term._evaluate(arithmetics))
        return sum

    def to_matrix(self: OperatorSum, dimensions: dict[int, int], **kwargs: SupportsComplex) -> NDArray[numpy.complexfloating]:
        """
        Arguments:
            dimensions: A dictionary that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the operator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The matrix representation of the operator expression in canonical order.
        """
        return self._evaluate(MatrixArithmetics(dimensions, **kwargs)).matrix

    def __str__(self: OperatorSum) -> str:
        return self._evaluate(PrettyPrint())

    def __mul__(self: OperatorSum, other: Any) -> OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == OperatorSum:
            return OperatorSum([self_term * other_term for self_term in self._terms for other_term in other._terms])
        return OperatorSum([self_term * other for self_term in self._terms])

    def __add__(self: OperatorSum, other: Any) -> OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented        
        elif type(other) == OperatorSum:
            return OperatorSum(self._terms + other._terms)
        return other + self # Operator addition is commutative.

    def __sub__(self: OperatorSum, other: Any) -> OperatorSum:
        return self + (-1 * other)

    # We only need to right-handle arithmetics with numbers, 
    # since everything else is covered by the left-hand arithmetics.

    def __rmul__(self: OperatorSum, other: Any) -> OperatorSum:
        if isinstance(other, (complex, float, int)):
            return OperatorSum([self_term * other for self_term in self._terms])
        return NotImplemented

    def __radd__(self: OperatorSum, other: Any) -> OperatorSum:
        if isinstance(other, (complex, float, int)):
            other_term = ProductOperator([operators.const(other)])
            return OperatorSum(self._terms + [other_term])
        return NotImplemented

    def __rsub__(self: OperatorSum, other: Any) -> OperatorSum:
        return (-1 * self) + other


class ProductOperator(OperatorSum):
    """
    Represents an operator expression consisting of a product of elementary
    and scalar operators. 
    """

    def __init__(self: ProductOperator, atomic_operators : list[ElementaryOperator | ScalarOperator] = []) -> None:
        """
        Instantiates an operator expression that represents a product of elementary
        or scalar operators.

        Arguments:
            atomic_operators: The operators of which to compute the product when 
                evaluating the operator expression.
        """
        if len(atomic_operators) == 0:
            atomic_operators = [operators.const(1)]
        self._operators = atomic_operators
        super().__init__([self])

    def __eq__(self: ProductOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is a ProductOperator consisting of the same elementary
            and scalar operators, and False otherwise.
        """
        return type(other) == ProductOperator and self._operators == other._operators

    @property
    def parameters(self: ProductOperator) -> dict[str, str]:
        """
        A dictionary that contains the documentation comment for each parameter 
        needed to evaluate the operator expression.
        """
        return _OperatorHelpers.aggregate_parameters([operator.parameters for operator in self._operators])

    def _evaluate(self: ProductOperator, arithmetics : OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function used for evaluating operator expressions and computing arbitrary values
        during evaluation. The value to be computed is defined by the OperatorArithmetics.
        The evaluation guarantees that addition and multiplication of two operators will only
        be called when both operators act on the same degrees of freedom, and the tensor product
        will only be computed if they act on different degrees of freedom. 
        """
        def padded_op(op: ElementaryOperator | ScalarOperator, degrees: list[int]):
            evaluated_ops = [] # Creating the tensor product with op being last is most efficient.
            for degree in degrees:
                if not degree in [d for d in op._degrees]:
                    evaluated_ops.append(operators.identity(degree)._evaluate(arithmetics))
            evaluated_ops.append(op._evaluate(arithmetics))
            padded = evaluated_ops[0]
            for value in evaluated_ops[1:]:
                padded = arithmetics.tensor(padded, value)
            return padded
        # Sorting the degrees to avoid unnecessary permutations during the padding.
        degrees = sorted(set([degree for op in self._operators for degree in op._degrees]))
        evaluated = padded_op(self._operators[0], degrees)
        for op in self._operators[1:]:
            if len(op._degrees) != 1 or op != operators.identity(op._degrees[0]):
                evaluated = arithmetics.mul(evaluated, padded_op(op, degrees))
        return evaluated

    def to_matrix(self: ProductOperator, dimensions: dict[int, int], **kwargs: SupportsComplex) -> NDArray[numpy.complexfloating]:
        """
        Arguments:
            dimensions: A dictionary that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the operator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The matrix representation of the operator expression in canonical order.
        """
        return self._evaluate(MatrixArithmetics(dimensions, **kwargs)).matrix

    def __str__(self: ProductOperator) -> str:
        return self._evaluate(PrettyPrint())

    def __mul__(self: ProductOperator, other: Any) -> ProductOperator | OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == ProductOperator:
            return ProductOperator(self._operators + other._operators)
        elif type(other) == OperatorSum: # Only create OperatorSum if needed.
            return OperatorSum([self]) * other
        elif isinstance(other, (complex, float, int)):
            return ProductOperator(self._operators + [operators.const(other)])
        return self * ProductOperator([other])

    def __add__(self: ProductOperator, other: Any) -> OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == ProductOperator:
            return OperatorSum([self, other])
        return OperatorSum([self]) + other

    def __sub__(self: ProductOperator, other: Any) -> OperatorSum:
        return self + (-1 * other)

    # We only need to right-handle arithmetics with numbers, 
    # since everything else is covered by the left-hand arithmetics.

    def __rmul__(self: ProductOperator, other: Any) -> ProductOperator:
        if isinstance(other, (complex, float, int)):
            return ProductOperator([operators.const(other)] + self._operators)
        return NotImplemented

    def __radd__(self: ProductOperator, other: Any) -> OperatorSum:
        return self + other # Operator addition is commutative.

    def __rsub__(self: ProductOperator, other: Any) -> OperatorSum:
        return (-1 * self) + other


class ElementaryOperator(ProductOperator):
    """
    Represents an elementary operator that acts on one or more degrees of freedom, 
    and cannot be further simplified to be the product or sum of other operators. 
    An elementary operator is defined as a function of zero or more complex-valued 
    parameters using the :func:`~dynamics.ElementaryOperator.define` class method.
    """

    _ops : dict[str, Callable[..., NDArray[numpy.complexfloating]]] = {}
    """
    Contains the generator for each defined ElementaryOperator.
    The generator takes a dictionary defining the dimensions for each degree of freedom,
    as well as keyword arguments for complex values.
    """
    _parameter_info : dict[str, dict[str, str]] = {}
    """
    Contains information about all required parameters for each ElementaryOperator.
    The dictionary keys are the parameter names, and the values are their documentation.
    """

    @classmethod
    def define(cls, op_id: str, expected_dimensions: list[int], create: Callable[..., NDArray[numpy.complexfloating]]) -> None:
        """
        Adds a generator function for an elementary operator with the given id to the
        class. After definition, an the defined elementary operator can be instantiated 
        by providing the operator id as well as the degree(s) of freedom that it acts on.
        
        An elementary operator is a parameterized object acting on certain degrees of 
        freedom. To evaluate an operator, for example to compute its matrix, the level, 
        that is the dimension, for each degree of freedom it acts on must be provided, 
        as well as all additional parameters. Additional parameters must be provided in 
        the form of keyword arguments.

        Note:
        The dimensions passed during operator evaluation are automatically validated 
        against the expected dimensions specified during definition - the `create` 
        function does not need to do this.

        Arguments:
            op_id: A string that uniquely identifies the defined operator.
            expected_dimensions: defines the number of levels, that is the dimension,
                for each degree of freedom in canonical (that is sorted) order. A 
                negative or zero value for one (or more) of the expected dimensions 
                indicates that the operator is defined for any dimension of the 
                corresponding degree of freedom.
            create: Takes any number of complex-valued arguments and returns the 
                matrix representing the operator in canonical order. If the matrix can
                be defined for any number of levels for one or more degree of freedom, 
                the `create` function must take an argument called `dimension` (or `dim`
                for short), if the operator acts on a single degree of freedom, and an
                argument called `dimensions` (or `dims` for short), if the operator acts 
                on multiple degrees of freedom.
        """
        if op_id in cls._ops:
            raise ValueError(f"an ElementaryOperator with id {op_id} already exists")
        elif len(expected_dimensions) == 0:
            raise ValueError(f"an ElementaryOperator needs to act on at least one degree " 
                             + "of freedom - use a ScalarOperator to define operators that "
                             + "do not act on any degrees of freedom")

        forwarded_as_kwarg = [["dimensions", "dims"], ["dimension", "dim"]]
        def with_dimension_check(creation: Callable, dimensions: list[int], **kwargs) -> NDArray[numpy.complexfloating]:
            if any([expected > 0 and dimensions[i] != expected for i, expected in enumerate(expected_dimensions)]):
                raise ValueError(f'no built-in operator {op_id} has been defined '\
                                 f'for {len(dimensions)} degree(s) of freedom with dimension(s) {dimensions}')
            # If the population of kwargs here is changed, adjust the filtering 
            # in the `parameters` property below.
            for forwarded in forwarded_as_kwarg[0]:
                kwargs[forwarded] = kwargs.get(forwarded, dimensions) # add if it does not exist
            if len(dimensions) == 1: 
                for forwarded in forwarded_as_kwarg[1]:
                    kwargs[forwarded] = kwargs.get(forwarded, dimensions[0]) # add if it does not exist
            creation_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(creation, **kwargs)
            evaluated = creation(*creation_args, **remaining_kwargs)
            if not isinstance(evaluated, numpy.ndarray):
                raise TypeError("operator evaluation must return a NDArray[complex]")
            return evaluated
        generator = lambda dimensions, **kwargs: with_dimension_check(create, dimensions, **kwargs)
        parameters, forwarded = {}, list([keyword for group in forwarded_as_kwarg for keyword in group])
        arg_spec = inspect.getfullargspec(create)
        for pname in arg_spec.args + arg_spec.kwonlyargs:
            if not pname in forwarded:
                parameters[pname] = _OperatorHelpers.parameter_docs(pname, create.__doc__)
        cls._ops[op_id] = generator
        cls._parameter_info[op_id] = parameters

    def __init__(self: ElementaryOperator, operator_id: str, degrees: list[int]) -> None:
        """
        Instantiates an elementary operator.

        Arguments:
            operator_id: The id of the operator as specified when it was defined.
            degrees: The degrees of freedom that the operator acts on.
        """
        if not operator_id in ElementaryOperator._ops:
            raise ValueError(f"no built-in operator '{operator_id}' has been defined")
        self._op_id = operator_id
        self._degrees = sorted(degrees) # Sorting so that order matches the elementary operation definition.
        super().__init__([self])

    def __eq__(self: ElementaryOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is an ElementaryOperator with the same operator id acting
            on the same degrees of freedom, and False otherwise.
        """
        return type(other) == ElementaryOperator and self._op_id == other._op_id and self._degrees == other._degrees

    @property
    def parameters(self: ElementaryOperator) -> dict[str, str]:
        """
        A dictionary that contains the documentation comment for each parameter 
        needed to evaluate the elementary operator.
        """
        return ElementaryOperator._parameter_info[self._op_id]

    def _evaluate(self: ElementaryOperator, arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function for consistency with other operator expressions.
        Invokes the `evaluate` method of the given OperatorArithmetics.
        """
        return arithmetics.evaluate(self)

    def to_matrix(self: ElementaryOperator, dimensions: dict[int, int], **kwargs: SupportsComplex) -> NDArray[numpy.complexfloating]:
        """
        Arguments:
            dimensions: A dictionary that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the operator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The matrix representation of the operator expression in canonical order.
        """
        missing_degrees = [degree not in dimensions for degree in self._degrees]
        if any(missing_degrees):
            raise ValueError(f'missing dimensions for degree(s) {[self._degrees[i] for i, x in enumerate(missing_degrees) if x]}')
        relevant_dimensions = [dimensions[d] for d in self._degrees]
        return ElementaryOperator._ops[self._op_id](relevant_dimensions, **kwargs)

    def __str__(self: ElementaryOperator) -> str:
        parameter_names = ", ".join(self.parameters)
        if parameter_names != "": parameter_names = f"({parameter_names})"
        return f"{self._op_id}{parameter_names}{self._degrees}"

    def __mul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == ElementaryOperator:
            return ProductOperator([self, other])
        return ProductOperator([self]) * other

    def __add__(self: ElementaryOperator, other: Any) -> OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == ElementaryOperator:
            op1 = ProductOperator([self])
            op2 = ProductOperator([other])
            return OperatorSum([op1, op2])
        return ProductOperator([self]) + other

    def __sub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + (-1 * other)

    # We only need to right-handle arithmetics with numbers, 
    # since everything else is covered by the left-hand arithmetics.

    def __rmul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        if isinstance(other, (complex, float, int)):
            return ProductOperator([self]) * other
        return NotImplemented

    def __radd__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + other # Operator addition is commutative.

    def __rsub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return (-1 * self) + other


class ScalarOperator(ProductOperator):
    """
    Represents a scalar operator defined as a function of zero or more 
    complex-valued parameters.
    """

    def __init__(self: ScalarOperator, generator: Callable[..., SupportsComplex]) -> None:
        """
        Instantiates a scalar operator.

        Arguments:
            generator: The value of the scalar operator as a function of its
                parameters. The generator may take any number of complex-valued
                arguments and must return a number. Each parameter must be passed
                as a keyword argument when evaluating the operator. 
        """
        self._degrees : list[int] = []
        self.generator = generator # The setter validates the generator and sets _parameter_info.
        super().__init__([self])

    def __eq__(self: ScalarOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is a ScalarOperator with the same generator,
            and False otherwise. The comparison does not evaluate the generator
            and may evaluate to False even if the two generators always produce the
            same values.
        """
        return type(other) == ScalarOperator and self._generator == other._generator

    @property
    def generator(self: ScalarOperator) -> Callable[..., SupportsComplex]:
        """
        The function that generates the value of the scalar operator. 
        The function can take any number of complex-valued arguments
        and returns a number.
        """
        return self._generator

    @generator.setter
    def generator(self: ScalarOperator, generator: Callable[..., SupportsComplex]) -> None:
        """
        Sets the generator function. The generator function must not take a 
        variable number of arguments (that is it must not take `*args`), and 
        its parameter names should be descriptive keywords and the parameters 
        should be documented following Google documentation comment style. 

        Setting the generator of a scalar operator updates its evaluated value
        in all operators that contain the scalar operator.
        """
        # A variable number of arguments (i.e. *args) cannot be supported
        # for generators; it would prevent proper argument handling while 
        # supporting additions and multiplication of all kinds of operators.
        arg_spec = inspect.getfullargspec(generator)
        if arg_spec.varargs is not None:
            raise ValueError(f"generator for a '{type(self).__name__}' must not take *args")
        param_descriptions = {}
        for arg_name in arg_spec.args + arg_spec.kwonlyargs:
            param_descriptions[arg_name] = _OperatorHelpers.parameter_docs(arg_name, generator.__doc__)
        # We need to create a lambda to retrieve information about what
        # parameters are required to invoke the generator, to ensure that
        # the information accurately captures any updates to the generators
        # of sub-operators, just like the lambda in the add/multiply below.    
        self._parameter_info = lambda: param_descriptions
        self._generator = generator

    @property
    def parameters(self: ScalarOperator) -> dict[str, str]:
        """
        A dictionary that contains the documentation comment for each parameter 
        needed to evaluate the generator.
        """
        return self._parameter_info()

    def _invoke(self: ScalarOperator, **kwargs: SupportsComplex) -> SupportsComplex:
        """
        Helper function that extracts the necessary arguments from the given keyword 
        arguments and invokes the generator. 
        """
        generator_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(self._generator, **kwargs)
        evaluated = self._generator(*generator_args, **remaining_kwargs)
        if not isinstance(evaluated, (complex, float, int)):
            raise TypeError(f"generator of {type(self).__name__} must return a number")
        return evaluated

    def _evaluate(self: ScalarOperator, arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function for consistency with other operator expressions.
        Invokes the `evaluate` method of the given OperatorArithmetics.
        """
        return arithmetics.evaluate(self)

    # The argument `dimensions` here is only passed for consistency with parent classes.
    def to_matrix(self: ScalarOperator, dimensions: dict[int, int] = {}, **kwargs: SupportsComplex) -> NDArray[numpy.complexfloating]:
        """
        Class method for consistency with other operator classes.
        Invokes the generator with the given keyword arguments.

        Arguments:
            dimensions: (unused, passed for consistency) 
                A dictionary that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the generator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The scalar value of the operator for the given keyword arguments.
        """
        return numpy.array([self._invoke(**kwargs)], dtype=numpy.complex128)

    def __str__(self: ScalarOperator) -> str:
        parameter_names = ", ".join(self.parameters)
        if parameter_names == "":
            return str(self._invoke())
        return f"{self._generator.__name__ or 'f'}({parameter_names})"

    def _compose(self: ScalarOperator, 
                 fct: Callable[[SupportsComplex], SupportsComplex], 
                 get_params: Optional[Callable[[], dict[str, str]]]) -> ScalarOperator:
        """
        Helper function to avoid duplicate code in the various arithmetic 
        operations supported on a ScalarOperator.
        """
        generator = lambda **kwargs: fct(self._invoke(**kwargs), **kwargs)
        operator = ScalarOperator(generator)
        if get_params is None:
            operator._parameter_info = self._parameter_info
        else:
            operator._parameter_info = lambda: _OperatorHelpers.aggregate_parameters([self._parameter_info(), get_params()])
        return operator

    def __pow__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value ** other
            return self._compose(fct, None)
        return NotImplemented

    def __mul__(self: ScalarOperator, other: Any) -> ScalarOperator | ProductOperator:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value * other._invoke(**kwargs)
            return self._compose(fct, other._parameter_info)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value * other
            return self._compose(fct, None)
        return ProductOperator([self]) * other
    
    def __truediv__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value / other
            return self._compose(fct, None)
        return NotImplemented

    def __add__(self: ScalarOperator, other: Any) -> ScalarOperator | OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, (complex, float, int))):
            return NotImplemented
        elif type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value + other._invoke(**kwargs)
            return self._compose(fct, other._parameter_info)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value + other
            return self._compose(fct, None)
        return ProductOperator([self]) + other

    def __sub__(self: ScalarOperator, other: Any) -> ScalarOperator | OperatorSum:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value - other
            return self._compose(fct, None)
        return ProductOperator([self]) - other

    # We only need to right-handle arithmetics with numbers, 
    # since everything else is covered by the left-hand arithmetics.

    def __rpow__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other ** value
            return self._compose(fct, None)
        return NotImplemented

    def __rmul__(self: ScalarOperator, other: Any) -> ProductOperator:
        return self * other # Scalar multiplication is commutative.

    def __rtruediv__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other / value
            return self._compose(fct, None)
        return NotImplemented

    def __radd__(self: ScalarOperator, other: Any) -> OperatorSum:
        return self + other # Operator addition is commutative.

    def __rsub__(self: ScalarOperator, other: Any) -> OperatorSum:
        return (-1 * self) + other


# Operators as defined here (watch out of differences in convention): 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html
class operators:

    class matrices:
        @staticmethod
        def _create(dimension: int) -> NDArray[numpy.complexfloating]: 
            return numpy.diag(numpy.sqrt(numpy.arange(1, dimension, dtype=numpy.complex128)), -1)
        @staticmethod
        def _annihilate(dimension: int) -> NDArray[numpy.complexfloating]: 
            return numpy.diag(numpy.sqrt(numpy.arange(1, dimension, dtype=numpy.complex128)), 1)
        @staticmethod
        def _position(dimension: int) -> NDArray[numpy.complexfloating]:
            return complex(0.5) * (operators.matrices._create(dimension) + operators.matrices._annihilate(dimension))
        @staticmethod
        def _momentum(dimension: int) -> NDArray[numpy.complexfloating]:
            return 0.5j * (operators.matrices._create(dimension) - operators.matrices._annihilate(dimension))
        @staticmethod
        def _displace(dimension: int, displacement: SupportsComplex) -> NDArray[numpy.complexfloating]:
            """Connects to the next available port.
            Args:
                displacement: Amplitude of the displacement operator.
                    See also https://en.wikipedia.org/wiki/Displacement_operator.
            """
            displacement = complex(displacement)
            term1 = displacement * operators.matrices._create(dimension)
            term2 = numpy.conjugate(displacement) * operators.matrices._annihilate(dimension)
            return scipy.linalg.expm(term1 - term2)
        @staticmethod
        def _squeeze(dimension: int, squeezing: SupportsComplex) -> NDArray[numpy.complexfloating]:
            """Connects to the next available port.
            Args:
                squeezing: Amplitude of the squeezing operator.
                    See also https://en.wikipedia.org/wiki/Squeeze_operator.
            """
            squeezing = complex(squeezing)
            term1 = numpy.conjugate(squeezing) * numpy.linalg.matrix_power(operators.matrices._annihilate(dimension), 2)
            term2 = squeezing * numpy.linalg.matrix_power(operators.matrices._create(dimension), 2)
            return scipy.linalg.expm(0.5 * (term1 - term2))

    ElementaryOperator.define("op_zero", [0], lambda dim: numpy.zeros((dim, dim), dtype=numpy.complex128))
    ElementaryOperator.define("op_identity", [0], lambda dim: numpy.diag(numpy.ones(dim, dtype=numpy.complex128)))
    ElementaryOperator.define("op_create", [0], matrices._create)
    ElementaryOperator.define("op_annihilate", [0], matrices._annihilate)
    ElementaryOperator.define("op_number", [0], lambda dim: numpy.diag(numpy.arange(dim, dtype=numpy.complex128)))
    ElementaryOperator.define("op_parity", [0], lambda dim: numpy.diag([(-1.+0j)**i for i in range(dim)]))
    ElementaryOperator.define("op_displace", [0], matrices._displace)
    ElementaryOperator.define("op_squeeze", [0], matrices._squeeze)
    ElementaryOperator.define("op_position", [0], matrices._position)
    ElementaryOperator.define("op_momentum", [0], matrices._momentum)

    @classmethod
    def const(cls, constant_value: SupportsComplex) -> ScalarOperator:
        return ScalarOperator(lambda: numpy.complex128(constant_value))
    @classmethod
    def zero(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_zero", [degree])
    @classmethod
    def identity(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_identity", [degree])
    @classmethod
    def create(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_create", [degree])
    @classmethod
    def annihilate(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_annihilate", [degree])
    @classmethod
    def number(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_number", [degree])
    @classmethod
    def parity(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_parity", [degree])
    @classmethod
    def displace(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_displace", [degree])
    @classmethod
    def squeeze(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_squeeze", [degree])
    @classmethod
    def position(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_position", [degree])
    @classmethod
    def momentum(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_momentum", [degree])

class pauli:
    ElementaryOperator.define("pauli_x", [2], lambda: numpy.array([[0,1],[1,0]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_y", [2], lambda: numpy.array([[0,1j],[-1j,0]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_z", [2], lambda: numpy.array([[1,0],[0,-1]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_i", [2], lambda: numpy.array([[1,0],[0,1]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_plus", [2], lambda: numpy.array([[0,0],[1,0]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_minus", [2], lambda: numpy.array([[0,1],[0,0]], dtype=numpy.complex128))

    @classmethod
    def x(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_x", [degree])
    @classmethod
    def y(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_y", [degree])
    @classmethod
    def z(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_z", [degree])
    @classmethod
    def i(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_i", [degree])
    @classmethod
    def plus(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_plus", [degree])
    @classmethod
    def minus(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_minus", [degree])


class Schedule:
    """
    Represents an iterator that produces all values needed for evaluating
    an operator expression at different time steps.
    """

    # The output type of the iterable steps must match the second argument of `get_value`.
    def __init__(self: Schedule, steps: Iterable[Any], parameters: Iterable[str], get_value: Callable[[str, Any], SupportsComplex]) -> None:
        """
        Creates a schedule for evaluating an operator expression at different steps.

        Arguments:
            steps: The sequence of steps in the schedule. A step is defined as a value 
                of arbitrary type.
            parameters: A sequence of strings representing the parameter names of an 
                operator expression.
            get_value: A function that takes the name of a parameter as well as an 
                additional value ("step") of arbitrary type as argument and returns the 
                complex value for that parameter at the given step.
        """
        self._iterator = iter(steps)
        self._parameters = parameters
        self._get_value = get_value
        self._current_step = None

    @property
    def current_step(self: Schedule) -> Optional[Any]:
        """
        The value of the step the Schedule (iterator) is currently at.
        """
        return self._current_step
    
    def __iter__(self: Schedule) -> Schedule:
        return self
        
    def __next__(self: Schedule) -> dict[str, SupportsComplex]:
        self._current_step = next(self._iterator)
        kwargs : dict[str, SupportsComplex] = {}
        for parameter in self._parameters:
            kwargs[parameter] = self._get_value(parameter, self._current_step)
        return kwargs


dims = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

print(f'pauliX(1): {pauli.x(1).to_matrix(dims)}')
print(f'pauliY(2): {pauli.y(2).to_matrix(dims)}')

print(f'pauliZ(0) * pauliZ(0): {(pauli.z(0) * pauli.z(0)).to_matrix(dims)}')
print(f'pauliZ(0) * pauliZ(1): {(pauli.z(0) * pauli.z(1)).to_matrix(dims)}')
print(f'pauliZ(0) * pauliY(1): {(pauli.z(0) * pauli.y(1)).to_matrix(dims)}')

op1 = ProductOperator([pauli.x(0), pauli.i(1)])
op2 = ProductOperator([pauli.i(0), pauli.x(1)])
print(f'pauliX(0) + pauliX(1): {op1.to_matrix(dims) + op2.to_matrix(dims)}')
op3 = ProductOperator([pauli.x(1), pauli.i(0)])
op4 = ProductOperator([pauli.i(1), pauli.x(0),])
print(f'pauliX(1) + pauliX(0): {op1.to_matrix(dims) + op2.to_matrix(dims)}')

print(f'pauliX(0) + pauliX(1): {(pauli.x(0) + pauli.x(1)).to_matrix(dims)}')
print(f'pauliX(0) * pauliX(1): {(pauli.x(0) * pauli.x(1)).to_matrix(dims)}')
print(f'pauliX(0) * pauliI(1) * pauliI(0) * pauliX(1): {(op1 * op2).to_matrix(dims)}')

print(f'pauliX(0) * pauliI(1): {op1.to_matrix(dims)}')
print(f'pauliI(0) * pauliX(1): {op2.to_matrix(dims)}')
print(f'pauliX(0) * pauliI(1) + pauliI(0) * pauliX(1): {(op1 + op2).to_matrix(dims)}')

op5 = pauli.x(0) * pauli.x(1)
op6 = pauli.z(0) * pauli.z(1)
print(f'pauliX(0) * pauliX(1): {op5.to_matrix(dims)}')
print(f'pauliZ(0) * pauliZ(1): {op6.to_matrix(dims)}')
print(f'pauliX(0) * pauliX(1) + pauliZ(0) * pauliZ(1): {(op5 + op6).to_matrix(dims)}')

op7 = pauli.x(0) + pauli.x(1)
op8 = pauli.z(0) + pauli.z(1)
print(f'pauliX(0) + pauliX(1): {op7.to_matrix(dims)}')
print(f'pauliZ(0) + pauliZ(1): {op8.to_matrix(dims)}')
print(f'pauliX(0) + pauliX(1) + pauliZ(0) + pauliZ(1): {(op7 + op8).to_matrix(dims)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(0) + pauliZ(1)): {(op7 * op8).to_matrix(dims)}')

print(f'pauliX(0) * (pauliZ(0) + pauliZ(1)): {(pauli.x(0) * op8).to_matrix(dims)}')
print(f'(pauliZ(0) + pauliZ(1)) * pauliX(0): {(op8 * pauli.x(0)).to_matrix(dims)}')

op9 = pauli.z(1) + pauli.z(2)
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {numpy.kron(op7.to_matrix(dims), pauli.i(2).to_matrix(dims))}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(op7 * pauli.i(2)).to_matrix(dims)}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(pauli.i(2) * op7).to_matrix(dims)}')
print(f'pauliI(0) * (pauliZ(1) + pauliZ(2)): {numpy.kron(pauli.i(0).to_matrix(dims), op9.to_matrix(dims))}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(op7 * op9).to_matrix(dims)}')

so0 = ScalarOperator(lambda: 1.0j)
print(f'Scalar op (t -> 1.0)(): {so0.to_matrix()}')

so1 = ScalarOperator(lambda t: t)
print(f'Scalar op (t -> t)(1.): {so1.to_matrix(t = 1.0)}')
print(f'Trivial prod op (t -> t)(1.): {(ProductOperator([so1])).to_matrix({}, t = 1.)}')
print(f'Trivial prod op (t -> t)(2.): {(ProductOperator([so1])).to_matrix({}, t = 2.)}')

print(f'(t -> t)(1j) * pauliX(0): {(so1 * pauli.x(0)).to_matrix(dims, t = 1j)}')
print(f'pauliX(0) * (t -> t)(1j): {(pauli.x(0) * so1).to_matrix(dims, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).to_matrix(dims, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).to_matrix(dims, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).to_matrix(dims, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).to_matrix(dims, t = 1j)}')
print(f'(t -> t)(2.) * (pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(so1 * op7 * op9).to_matrix(dims, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (t -> t)(2.) * (pauliZ(1) + pauliZ(2)): {(op7 * so1 * op9).to_matrix(dims, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)) * (t -> t)(2.): {(op7 * op9 * so1).to_matrix(dims, t = 2.)}')

op10 = so1 * pauli.x(0)
so1.generator = lambda t: 1./t
print(f'(t -> 1/t)(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')
so1_gen2 = so1.generator
so1.generator = lambda t: so1_gen2(2*t)
print(f'(t -> 1/(2t))(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')
so1.generator = lambda t: so1_gen2(t)
print(f'(t -> 1/t)(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')

so2 = ScalarOperator(lambda t: t**2)
op11 = pauli.z(1) * so2
print(f'pauliZ(0) * (t -> t^2)(2.): {op11.to_matrix(dims, t = 2.)}')

so3 = ScalarOperator(lambda t: 1./t)
so4 = ScalarOperator(lambda t: t**2)
print(f'((t -> 1/t) * (t -> t^2))(2.): {(so3 * so4).to_matrix(t = 2.)}')
so5 = so3 + so4
so3.generator = lambda field: 1./field
print(f'((f -> 1/f) + (t -> t^2))(f=2, t=1.): {so5.to_matrix(t = 1., field = 2)}')

def generator(field, **kwargs):
    print(f'generator got kwargs: {kwargs}')
    return field

so3.generator = generator
print(f'((f -> f) + (t -> t^2))(f=3, t=2): {so5.to_matrix(field = 3, t = 2, dummy = 10)}')

so6 = ScalarOperator(lambda foo, *, bar: foo * bar)
print(f'((f,t) -> f*t)(f=3, t=2): {so6.to_matrix(foo = 3, bar = 2, dummy = 10)}')
so7 = ScalarOperator(lambda foo, *, bar, **kwargs: foo * bar)
print(f'((f,t) -> f*t)(f=3, t=2): {so6.to_matrix(foo = 3, bar = 2, dummy = 10)}')

def get_parameter_value(parameter_name: str, time: float):
    match parameter_name:
        case "foo": return time
        case "bar": return 2 * time
        case _: raise NotImplementedError(f'No value defined for parameter {parameter_name}.')

schedule = Schedule([0.0, 0.5, 1.0], so6.parameters, get_parameter_value)
for parameters in schedule:
    print(f'step {schedule.current_step}')
    print(f'((f,t) -> f*t)({parameters}): {so6.to_matrix({}, **parameters)}')

print(f'(pauliX(0) + i*pauliY(0))/2: {0.5 * (pauli.x(0) + operators.const(1j) * pauli.y(0)).to_matrix(dims)}')
print(f'pauli+(0): {pauli.plus(0).to_matrix(dims)}')
print(f'(pauliX(0) - i*pauliY(0))/2: {0.5 * (pauli.x(0) - operators.const(1j) * pauli.y(0)).to_matrix(dims)}')
print(f'pauli-(0): {pauli.minus(0).to_matrix(dims)}')

op12 = operators.squeeze(0) + operators.displace(0)
print(f'create<3>(0): {operators.create(0).to_matrix({0:3})}')
print(f'annihilate<3>(0): {operators.annihilate(0).to_matrix({0:3})}')
print(f'squeeze<3>(0)[squeezing = 0.5]: {operators.squeeze(0).to_matrix({0:3}, squeezing=0.5)}')
print(f'displace<3>(0)[displacement = 0.5]: {operators.displace(0).to_matrix({0:3}, displacement=0.5)}')
print(f'(squeeze<3>(0) + displace<3>(0))[squeezing = 0.5, displacement = 0.5]: {op12.to_matrix({0:3}, displacement=0.5, squeezing=0.5)}')
print(f'squeeze<4>(0)[squeezing = 0.5]: {operators.squeeze(0).to_matrix({0:4}, squeezing=0.5)}')
print(f'displace<4>(0)[displacement = 0.5]: {operators.displace(0).to_matrix({0:4}, displacement=0.5)}')
print(f'(squeeze<4>(0) + displace<4>(0))[squeezing = 0.5, displacement = 0.5]: {op12.to_matrix({0:4}, displacement=0.5, squeezing=0.5)}')

so8 = ScalarOperator(lambda my_param: my_param - 1)
so9 = so7 * so8
print(f'parameter descriptions: {operators.squeeze(0).parameters}')
print(f'parameter descriptions: {op12.parameters}')
print(f'parameter descriptions: {(so7 + so8).parameters}')
print(f'parameter descriptions: {so9.parameters}')
so7.generator = lambda new_parameter: 1.0
print(f'parameter descriptions: {so9.parameters}')
so9.generator = lambda reset: reset
print(f'parameter descriptions: {so9.parameters}')

def all_zero(sure, args):
    """Some args documentation.
    Args:

      sure (:obj:`int`, optional): my docs for sure
      args: Description of `args`. Multiple
            lines are supported.
    Returns:
      Something that for sure is correct.
    """
    if sure: return 0
    else: return 1

print(f'parameter descriptions: {(ScalarOperator(all_zero)).parameters}')

scop = operators.const(2)
elop = operators.identity(1)
print(f"arithmetics: {scop.to_matrix(dims)}")
print(f"arithmetics: {elop.to_matrix(dims)}")
print(f"arithmetics: {(scop * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop + elop).to_matrix(dims)}")
print(f"arithmetics: {(elop + scop).to_matrix(dims)}")
print(f"arithmetics: {(scop - elop).to_matrix(dims)}")
print(f"arithmetics: {(elop - scop).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop * (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) + scop).to_matrix(dims)}")
print(f"arithmetics: {(scop + (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) + elop).to_matrix(dims)}")
print(f"arithmetics: {(elop + (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) - scop).to_matrix(dims)}")
print(f"arithmetics: {(scop - (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) - elop).to_matrix(dims)}")
print(f"arithmetics: {(elop - (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop * (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop * (scop - elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * (scop - elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) + scop).to_matrix(dims)}")
print(f"arithmetics: {(scop + (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) + elop).to_matrix(dims)}")
print(f"arithmetics: {(elop + (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) - scop).to_matrix(dims)}")
print(f"arithmetics: {(scop - (scop - elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) - elop).to_matrix(dims)}")
print(f"arithmetics: {(elop - (scop - elop)).to_matrix(dims)}")

opprod = operators.create(0) * operators.annihilate(0)
opsum = operators.create(0) + operators.annihilate(0)
for arith in [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]:
    print(f"testing {arith} for ScalarOperator")
    print(f"arithmetics: {arith(scop, 2).to_matrix(dims)}")
    print(f"arithmetics: {arith(scop, 2.5).to_matrix(dims)}")
    print(f"arithmetics: {arith(scop, 2j).to_matrix(dims)}")
    print(f"arithmetics: {arith(2, scop).to_matrix(dims)}")
    print(f"arithmetics: {arith(2.5, scop).to_matrix(dims)}")
    print(f"arithmetics: {arith(2j, scop).to_matrix(dims)}")

for op in [elop, opprod, opsum]:
    for arith in [operator.add, operator.sub, operator.mul]:
        print(f"testing {arith} for {type(op)}")
        print(f"arithmetics: {arith(op, 2).to_matrix(dims)}")
        print(f"arithmetics: {arith(op, 2.5).to_matrix(dims)}")
        print(f"arithmetics: {arith(op, 2j).to_matrix(dims)}")
        print(f"arithmetics: {arith(2, op).to_matrix(dims)}")
        print(f"arithmetics: {arith(2.5, op).to_matrix(dims)}")
        print(f"arithmetics: {arith(2j, op).to_matrix(dims)}")

print(operators.const(2))
print(ScalarOperator(lambda alpha: 2*alpha))
print(ScalarOperator(all_zero))
print(pauli.x(0))
print(2 * pauli.x(0))
print(pauli.x(0) + 2)
print(operators.squeeze(0))
print(operators.squeeze(0) * operators.displace(1))
print(operators.squeeze(0) + operators.displace(1) * 5)
print(pauli.x(0) - 2)
print(pauli.x(0) - pauli.y(1))

