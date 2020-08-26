from numpy import prod, arange, array, int8, intp, empty

from hypothesis import given, example
from hypothesis.strategies import one_of, integers

from pytest import raises

from .helpers import integer_arrays, short_shapes, check_same, assert_equal

from ..integer import Integer
from ..integerarray import IntegerArray

def test_integerarray_constructor():
    raises(ValueError, lambda: IntegerArray([0], shape=(1,)))
    raises(ValueError, lambda: IntegerArray([], shape=(1,)))
    raises(TypeError, lambda: IntegerArray([False]))
    raises(TypeError, lambda: IntegerArray(array(0.0)))
    raises(TypeError, lambda: IntegerArray((1,)))
    idx = IntegerArray(array([0], dtype=int8))
    assert_equal(idx.array, array([0], dtype=intp))

    idx = IntegerArray([], shape=(0, 1))
    assert_equal(idx.array, empty((0, 1), dtype=intp))

    # Make sure the underlying array is immutable
    idx = IntegerArray([1, 2])
    with raises(ValueError):
        idx.array[0] = 0
    assert_equal(idx.array, array([1, 2], dtype=intp))

    # Make sure the underlying array is copied
    a = array([1, 2])
    idx = IntegerArray(a)
    a[0] = 0
    assert idx == IntegerArray([1, 2])

@given(integer_arrays, short_shapes)
def test_integerarray_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)

@given(integer_arrays, one_of(short_shapes, integers(0, 10)))
def test_integerarray_reduce_no_shape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = IntegerArray(idx)

    check_same(a, index.raw, ndindex_func=lambda a, x: a[x.reduce().raw])

@example(array([2, 0]), (1, 0))
@example(array(0), 1)
@example(array([], dtype=intp), 0)
@given(integer_arrays, one_of(short_shapes, integers(0, 10)))
def test_integerarray_reduce_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = IntegerArray(idx)

    check_same(a, index.raw, ndindex_func=lambda a, x: a[x.reduce(shape).raw])

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

@example([], (1,))
@example([0], (1, 0))
@given(integer_arrays, one_of(short_shapes, integers(0, 10)))
def test_integerarray_isempty_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = IntegerArray(idx)


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
