from hypothesis import assume, given, settings, target, Verbosity
from hypothesis.strategies import (integers, composite, recursive, one_of,
lists, just, builds)

from sympy import Add, Mul, Min, Max, Mod, symbols, oo, Integer


@composite
def apply(draw, func, strategy):
    # Note, this assumes that func.nargs has a range of values with no gaps,
    # which is true for virtually all SymPy functions.
    def as_int(x):
        if x == oo:
            return None
        if x is not None:
            x = int(x)
        return x
    if hasattr(func, 'nargs'):
        min_size = as_int(func.nargs.inf)
        max_size = as_int(func.nargs.sup)
    else:
        # Assume it takes any number of args like Add or Mul
        min_size = 0
        max_size = None
    args = draw(lists(strategy, min_size=min_size,
                      max_size=max_size))
    try:
        return func(*args)
    except (ZeroDivisionError, ValueError, TypeError):
        assume(False)

def variables(nargs):
    return symbols('x:%d' % nargs)

def generate_function_strategy(nargs):
    vars = [just(i) for i in variables(nargs)]

    # This should be ordered from simplest to more complicated
    base = one_of(
        *vars,
        builds(Integer, integers(-1, 2)),
    )

    def extend(children):
        return one_of(
            apply(Add, children),
            apply(Mul, children),
            apply(Max, children),
            apply(Min, children),
            apply(Mod, children),
        )

    return recursive(base, extend, max_leaves=4)

class FoundExample(Exception):
    pass

def guess(inputs, outputs, max_examples=1000, verbose=False, use_target=True):
    """
    Guess a function that takes inputs and returns outputs

    Inputs should be a list of lists representing the input parameters, and
    outputs should be a list of outputs representing the output for each
    input.
    """
    if len({len(i) for i in inputs}) != 1:
        raise ValueError("All inputs should have the same length")
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs should be the same length")

    nargs = len(inputs[0])
    vars = variables(nargs)

    verbosity = Verbosity.verbose if verbose else Verbosity.normal

    seen = []
    @settings(max_examples=max_examples, verbosity=verbosity)
    @given(generate_function_strategy(nargs))
    def check_value(expr):
        assume(expr not in seen)
        error = 0
        try:
            for i, o in zip(inputs, outputs):
                val = expr.subs(zip(vars, i))
                assert val.is_Number
                e = abs(val - o)
                if not e.is_finite:
                    e = 1
                error += abs(val - o)
                if not use_target and error:
                    return
            if error == 0:
                raise FoundExample(expr)
            if use_target:
                target(-error)
        except (ZeroDivisionError, ValueError, TypeError):
            return

    return check_value()
