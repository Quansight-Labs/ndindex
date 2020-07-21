from numpy import prod, arange

from hypothesis import given

from .helpers import integer_arrays, shapes, check_same

@given(integer_arrays, shapes)
def test_integer_array_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)
