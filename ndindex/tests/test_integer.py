from numpy import arange, int64

from pytest import raises

from hypothesis import given
from hypothesis.strategies import integers, one_of

from ..integer import Integer
from ..ndindex import ndindex
from ..tuple import Tuple
from .helpers import check_same, ints, prod, shapes


def test_integer_args():
    zero = Integer(0)
    assert zero.raw == 0
    idx = Integer(int64(0))
    assert idx == zero
    assert idx.raw == 0
    assert isinstance(idx.raw, int)
    assert Integer(zero) == zero


def test_integer_exhaustive():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i)


@given(ints(), integers(5, 100))
def test_integer_hypothesis(i, size):
    a = arange(size)
    check_same(a, i)


def test_integer_len_exhaustive():
    for i in range(-12, 12):
        idx = Integer(i)
        assert len(idx) == 1


@given(ints())
def test_integer_len_hypothesis(i):
    idx = Integer(i)
    assert len(idx) == 1


def test_integer_reduce_exhaustive():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i, func=lambda x: x.reduce((10,)))

        try:
            reduced = Integer(i).reduce(10)
        except IndexError:
            pass
        else:
            assert reduced.raw >= 0


@given(ints(), shapes)
def test_integer_reduce_hypothesis(i, shape):
    a = arange(prod(shape)).reshape(shape)
    # The axis argument is tested implicitly in the Tuple.reduce test. It is
    # difficult to test here because we would have to pass in a Tuple to
    # check_same.
    check_same(a, i, func=lambda x: x.reduce(shape))

    try:
        reduced = Integer(i).reduce(shape)
    except IndexError:
        pass
    else:
        assert reduced.raw >= 0

def test_integer_reduce_no_shape_exhaustive():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i, func=lambda x: x.reduce())

@given(ints(), shapes)
def test_integer_reduce_no_shape_hypothesis(i, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, i, func=lambda x: x.reduce())

def test_integer_newshape_exhaustive():
    # The axis argument is tested implicitly in Tuple.newshape
    shape = 5
    a = arange(shape)

    for i in range(-10, 10):
        def assert_equal(x, y):
            newshape = ndindex(i).newshape(shape)
            assert x.shape == y.shape == newshape

        # Call newshape so we can see if any exceptions match
        def func(i):
            i.newshape(shape)
            return i

        check_same(a, i, func=func, assert_equal=assert_equal)

@given(ints(), one_of(shapes, integers(0, 10)))
def test_integer_newshape_hypothesis(i, shape):
    # The axis argument is tested implicitly in Tuple.newshape
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    def assert_equal(x, y):
        newshape = ndindex(i).newshape(shape)
        assert x.shape == y.shape == newshape

    # Call newshape so we can see if any exceptions match
    def func(i):
        i.newshape(shape)
        return i

    check_same(a, i, func=func, assert_equal=assert_equal)

def test_integer_newshape_ndindex_input():
    raises(TypeError, lambda: Integer(1).newshape(Tuple(2, 1)))
    raises(TypeError, lambda: Integer(1).newshape(Integer(2)))

def test_integer_newshape_small_shape():
    raises(IndexError, lambda: Integer(6).newshape(2))
    raises(IndexError, lambda: Integer(6).newshape((8, 4), axis=1))
    raises(IndexError, lambda: Integer(6).newshape((4, 4)))

def test_integer_newshape_wrong_axis():
    raises(IndexError, lambda: Integer(6).newshape(2, axis=1))
    raises(IndexError, lambda: Integer(6).newshape((4, 2), axis=2))
