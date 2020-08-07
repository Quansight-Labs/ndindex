from numpy import intp, array

from pytest import raises

from ..array import ArrayIndex
from ..integerarray import IntegerArray

from .helpers import assert_equal

# Everything else is tested in the subclasses

def test_ArrayIndex():
    raises(TypeError, lambda: ArrayIndex([]))

def test_attributes():
    a = array([[0, 1], [1, 0]])

    idx = IntegerArray(a)
    assert_equal(idx.array, array(a, dtype=intp))
    assert idx.dtype == intp
    assert idx.ndim == a.ndim == 2
    assert idx.shape == a.shape == (2, 2)
    assert idx.size == a.size == 4
