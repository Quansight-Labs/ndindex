from numpy import prod, arange, array, bool_, empty, full

from hypothesis import given, example
from hypothesis.strategies import one_of, integers

from pytest import raises

from .helpers import boolean_arrays, shapes, check_same, assert_equal

from ..booleanarray import BooleanArray

def test_booleanarray_constructor():
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
def test_booleanarray_hypothesis(idx, shape):
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

@example(full((1, 9), True), (3, 3))
@example(full((1, 9), False), (3, 3))
@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_reduce_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = BooleanArray(idx)

    if (index.count_nonzero == 0
        and prod(a.shape) == prod(index.shape) not in [0, 1]
        and len(a.shape) == len(index.shape)):
        # NumPy currently allows this case, due to a bug: (see
        # https://github.com/numpy/numpy/issues/16997 and
        # https://github.com/numpy/numpy/pull/17010), but we disallow it.
        with raises(IndexError, match=r"boolean index did not match indexed "
                    r"array along dimension \d+; dimension is \d+ but "
                    r"corresponding boolean dimension is \d+"):
            index.reduce(shape)
        return

    check_same(a, index.raw, func=lambda x: x.reduce(shape))

    try:
        reduced = index.reduce(shape)
    except IndexError:
        pass
    else:
        # At present, reduce() always returns the same index if it doesn't
        # give an IndexError
        assert reduced == index

@example(full((1, 9), False), (3, 3))
@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_newshape_hypothesis(idx, shape):
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

    index = BooleanArray(idx)
    if (index.count_nonzero == 0
        and prod(a.shape) == prod(index.shape) not in [0, 1]
        and len(a.shape) == len(index.shape)):
        # NumPy currently allows this case, due to a bug: (see
        # https://github.com/numpy/numpy/issues/16997 and
        # https://github.com/numpy/numpy/pull/17010), but we disallow it.
        with raises(IndexError, match=r"boolean index did not match indexed "
                    r"array along dimension \d+; dimension is \d+ but "
                    r"corresponding boolean dimension is \d+"):
            index.reduce(shape)
        return

    check_same(a, idx, func=func, assert_equal=assert_equal)


@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_isempty_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = BooleanArray(idx)

    # Call isempty to see if the exceptions are the same
    def func(index):
        index.isempty(shape)
        return index

    def assert_equal(a_raw, a_idx):
        isempty = index.isempty()
        isempty_shape = index.isempty(shape)

        aempty = (a_raw.size == 0)
        assert aempty == (a_idx.size == 0)

        # If isempty is true with no shape it should be true for a specific
        # shape. The converse is not true because the indexed array could be
        # empty.
        if isempty:
            assert isempty_shape

        # isempty() should always give the correct result for a specific
        # array after reduction
        assert isempty_shape == aempty, (index, shape)

    check_same(a, idx, func=func, assert_equal=assert_equal)
