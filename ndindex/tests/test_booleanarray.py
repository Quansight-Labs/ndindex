from numpy import prod, arange, array, bool_, empty

from hypothesis import given
from hypothesis.strategies import one_of, integers

from pytest import raises

from .helpers import boolean_arrays, shapes, check_same, assert_equal

from ..booleanarray import BooleanArray

def test_boolean_array_constructor():
    raises(ValueError, lambda: BooleanArray([False], shape=(1,)))
    raises(ValueError, lambda: BooleanArray([], shape=(1,)))
    raises(TypeError, lambda: BooleanArray([0]))
    raises(TypeError, lambda: BooleanArray(array(0.0)))
    raises(TypeError, lambda: BooleanArray((True,)))
    idx = BooleanArray(array([True], dtype=bool_))
    assert_equal(idx.array, array([True], dtype=bool_))

    idx = BooleanArray([], shape=(0, 1))
    assert_equal(idx.array, empty((0, 1), dtype=bool_))

    # Make sure the underlying array is immutable
    idx = BooleanArray([True])
    with raises(ValueError):
        idx.array[0] = False
    assert_equal(idx.array, array([True], dtype=bool_))

    # Make sure the underlying array is copied
    a = array([True, False])
    idx = BooleanArray(a)
    a[0] = False
    assert idx == BooleanArray([True, False])

@given(boolean_arrays, shapes)
def test_boolean_array_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)

@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_reduce_no_shape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = BooleanArray(idx)

    check_same(a, index.raw, func=lambda x: x.reduce())

@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_reduce_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = BooleanArray(idx)

    check_same(a, index.raw, func=lambda x: x.reduce(shape))

    try:
        reduced = index.reduce(shape)
    except IndexError:
        pass
    else:
        # At present, reduce() always returns the same index if it doesn't
        # give an IndexError
        assert reduced == index

@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_boolean_array_newshape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    def assert_equal(x, y):
        newshape = BooleanArray(idx).newshape(shape)
        assert x.shape == y.shape == newshape

    # Call newshape so we can see if any exceptions match
    def func(idx):
        idx.newshape(shape)
        return idx

    check_same(a, idx, func=func, assert_equal=assert_equal)
