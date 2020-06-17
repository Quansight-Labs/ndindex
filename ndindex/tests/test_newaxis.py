from numpy import arange, newaxis

from hypothesis import given

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
