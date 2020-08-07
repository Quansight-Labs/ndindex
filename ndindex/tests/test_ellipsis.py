from numpy import arange

from hypothesis import given
from hypothesis.strategies import one_of, integers

from pytest import raises

from ..ndindex import ndindex
from ..tuple import Tuple
from ..integer import Integer
from ..ellipsis import ellipsis
from .helpers import check_same, prod, shapes, ellipses

def test_ellipsis_exhaustive():
    for n in range(10):
        a = arange(n)
    check_same(a, ...)

@given(ellipses(), shapes)
def test_ellipsis_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)

def test_ellipsis_reduce_exhaustive():
    for n in range(10):
        a = arange(n)
        check_same(a, ..., func=lambda x: x.reduce((n,)))

@given(ellipses(), shapes)
def test_ellipsis_reduce_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, func=lambda x: x.reduce(shape))

def test_ellipsis_reduce_no_shape_exhaustive():
    for n in range(10):
        a = arange(n)
        check_same(a, ..., func=lambda x: x.reduce())

@given(ellipses(), shapes)
def test_ellipsis_reduce_no_shape_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, func=lambda x: x.reduce())

@given(ellipses(), one_of(shapes, integers(0, 10)))
def test_ellipsis_newshape_hypothesis(idx, shape):
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

def test_ellipsis_newshape_ndindex_input():
    raises(TypeError, lambda: ellipsis().newshape(Tuple(2, 1)))
    raises(TypeError, lambda: ellipsis().newshape(Integer(2)))

@given(ellipses(), one_of(shapes, integers(0, 10)))
def test_ellipsis_isempty_hypothesis(idx, shape):
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

        # Since idx is an ellipsis, it should never be unconditionally empty
        assert not isempty

        # isempty() should always give the correct result for a specific
        # array after reduction
        assert E.isempty(shape) == aempty, (E, shape)

    check_same(a, idx, func=func, assert_equal=assert_equal)
