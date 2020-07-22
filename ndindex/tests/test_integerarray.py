from numpy import prod, arange

from hypothesis import given
from hypothesis.strategies import one_of, integers

from .helpers import integer_arrays, shapes, check_same

from ..integer import Integer
from ..integerarray import IntegerArray

@given(integer_arrays, shapes)
def test_integer_array_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)

@given(integer_arrays, one_of(shapes, integers(0, 10)))
def test_integerarray_reduce_no_shape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = IntegerArray(idx)

    check_same(a, index.raw, func=lambda x: x.reduce())

@given(integer_arrays, one_of(shapes, integers(0, 10)))
def test_integerarray_reduce_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = IntegerArray(idx)

    check_same(a, index.raw, func=lambda x: x.reduce(shape))

    try:
        reduced = index.reduce(shape)
    except IndexError:
        pass
    else:
        if isinstance(reduced, Integer):
            assert reduced.raw >= 0
        else:
            assert isinstance(reduced, IntegerArray)
            assert (reduced.raw >= 0).all()
