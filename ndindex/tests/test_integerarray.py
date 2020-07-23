from numpy import prod, arange, array, int8, intp, empty

from hypothesis import given, example
from hypothesis.strategies import one_of, integers

from pytest import raises

from .helpers import integer_arrays, shapes, check_same, assert_equal

from ..integer import Integer
from ..integerarray import IntegerArray

def test_integer_array_constructor():
    raises(ValueError, lambda: IntegerArray([0], shape=(1,)))
    raises(ValueError, lambda: IntegerArray([], shape=(1,)))
    raises(TypeError, lambda: IntegerArray([False]))
    raises(TypeError, lambda: IntegerArray(array(0.0)))
    raises(TypeError, lambda: IntegerArray((1,)))
    idx = IntegerArray(array([0], dtype=int8))
    assert_equal(idx.array, array([0], dtype=intp))

    idx = IntegerArray([], shape=(0, 1))
    assert_equal(idx.array, empty((0, 1), dtype=intp))

    idx = IntegerArray([1, 2])
    with raises(ValueError):
        idx.array[0] = 0

    a = array([1, 2])
    idx = IntegerArray(a)
    a[0] = 0
    assert idx == IntegerArray([1, 2])

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

@example(array(0), 1)
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

@given(integer_arrays, one_of(shapes, integers(0, 10)))
def test_integer_array_newshape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    def assert_equal(x, y):
        newshape = IntegerArray(idx).newshape(shape)
        assert x.shape == y.shape == newshape

    # Call newshape so we can see if any exceptions match
    def func(idx):
        idx.newshape(shape)
        return idx

    check_same(a, idx, func=func, assert_equal=assert_equal)
