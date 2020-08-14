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

    check_same(a, index.raw, ndindex_func=lambda a, x: a[x.reduce().raw])

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
        and a.shape != index.shape
        and prod(a.shape) == prod(index.shape)
        and any(i != 0 and i != j for i, j in zip(index.shape, a.shape))
        and len(a.shape) == len(index.shape)):
        # NumPy currently allows this case, due to a bug: (see
        # https://github.com/numpy/numpy/issues/16997 and
        # https://github.com/numpy/numpy/pull/17010), but we disallow it.
        with raises(IndexError, match=r"boolean index did not match indexed "
                    r"array along dimension \d+; dimension is \d+ but "
                    r"corresponding boolean dimension is \d+"):
            index.reduce(shape)
        # Make sure this really is one of the cases NumPy lets through. Remove
        # this once a version of NumPy is released with the above fix.
        a[index.raw]
        return

    check_same(a, index.raw, ndindex_func=lambda a, x: a[x.reduce(shape).raw])

    try:
        reduced = index.reduce(shape)
    except IndexError:
        pass
    else:
        # At present, reduce() always returns the same index if it doesn't
        # give an IndexError
        assert reduced == index

@example(array([[[True], [False]]]), (1, 1, 2))
@example(full((1, 9), False), (3, 3))
@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_newshape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    def raw_func(a, idx):
        return a[idx].shape

    def ndindex_func(a, index):
        return index.newshape(shape)

    def assert_equal(raw_shape, newshape):
        assert raw_shape == newshape

    index = BooleanArray(idx)
    if (index.count_nonzero == 0
        and a.shape != index.shape
        and prod(a.shape) == prod(index.shape)
        and any(i != 0 and i != j for i, j in zip(index.shape, a.shape))
        and len(a.shape) == len(index.shape)):
        # NumPy currently allows this case, due to a bug: (see
        # https://github.com/numpy/numpy/issues/16997 and
        # https://github.com/numpy/numpy/pull/17010), but we disallow it.
        with raises(IndexError, match=r"boolean index did not match indexed "
                    r"array along dimension \d+; dimension is \d+ but "
                    r"corresponding boolean dimension is \d+"):
            index.newshape(shape)
        # Make sure this really is one of the cases NumPy lets through. Remove
        # this once a version of NumPy is released with the above fix.
        a[index.raw]
        return

    check_same(a, idx, raw_func=raw_func, ndindex_func=ndindex_func,
               assert_equal=assert_equal, same_exception=False)


@given(boolean_arrays, one_of(shapes, integers(0, 10)))
def test_booleanarray_isempty_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = BooleanArray(idx)

    def raw_func(a, idx):
        return a[idx].size == 0

    def ndindex_func(a, index):
        return index.isempty(), index.isempty(shape)

    def assert_equal(raw_empty, ndindex_empty):
        isempty, isempty_shape = ndindex_empty

        # If isempty is True then a[t] should be empty
        if isempty:
            assert raw_empty, (index, shape)
        # We cannot test the converse with hypothesis. isempty may be False
        # but a[idx] could still be empty for this specific a (e.g., if a is
        # already itself empty).

        # If isempty is true with no shape it should be true for a specific
        # shape. The converse is not true because the indexed array could be
        # empty.
        if isempty:
            assert isempty_shape, (index, shape)

        # isempty() should always give the correct result for a specific
        # array after reduction
        assert isempty_shape == raw_empty, (index, shape)

    check_same(a, idx, raw_func=raw_func, ndindex_func=ndindex_func,
               assert_equal=assert_equal, same_exception=False)
