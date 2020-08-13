from numpy import arange, newaxis

from pytest import raises

from hypothesis import given
from hypothesis.strategies import one_of, integers

from ..ndindex import ndindex
from ..integer import Integer
from ..tuple import Tuple
from ..newaxis import Newaxis
from .helpers import check_same, prod, shapes, newaxes

def test_newaxis_exhaustive():
    for n in range(10):
        a = arange(n)
    check_same(a, newaxis)


@given(newaxes(), shapes)
def test_newaxis_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)


def test_newaxis_reduce_exhaustive():
    for n in range(10):
        a = arange(n)
        check_same(a, newaxis, func=lambda x: x.reduce((n,)))


@given(newaxes(), shapes)
def test_newaxis_reduce_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, func=lambda x: x.reduce(shape))


def test_newaxis_reduce_no_shape_exhaustive():
    for n in range(10):
        a = arange(n)
        check_same(a, newaxis, func=lambda x: x.reduce())

@given(newaxes(), shapes)
def test_newaxis_reduce_no_shape_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, func=lambda x: x.reduce())

@given(newaxes(), one_of(shapes, integers(0, 10)))
def test_newaxis_newshape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = ndindex(idx)

    # Call newshape so we can see if any exceptions match
    def func(t):
        t.newshape(shape)
        return t

    def assert_equal(x, y):
        newshape = index.newshape(shape)
        assert x.shape == y.shape == newshape

    check_same(a, idx, func=func, assert_equal=assert_equal)

def test_newaxis_newshape_ndindex_input():
    raises(TypeError, lambda: Newaxis().newshape(Tuple(2, 1)))
    raises(TypeError, lambda: Newaxis().newshape(Integer(2)))

@given(newaxes(), one_of(shapes, integers(0, 10)))
def test_newaxis_isempty_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    E = ndindex(idx)

    # Call isempty to see if the exceptions are the same
    def func(E):
        E.isempty(shape)
        return E

    def assert_equal(a_raw, a_idx):
        isempty = E.isempty()

        aempty = (a_raw.size == 0)
        assert aempty == (a_idx.size == 0)

        # Since idx is a newaxis, it should never be unconditionally empty
        assert not isempty

        # isempty() should always give the correct result for a specific
        # array after reduction
        assert E.isempty(shape) == aempty, (E, shape)

    check_same(a, idx, func=func, assert_equal=assert_equal)
