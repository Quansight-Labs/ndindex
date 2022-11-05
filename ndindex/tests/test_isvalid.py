from pytest import raises

from hypothesis import given

from numpy import arange, prod

from ..ndindex import ndindex
from .helpers import ndindices, shapes

@given(ndindices, shapes)
def test_isvalid_hypothesis(idx, shape):
    index = ndindex(idx)

    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    valid = index.isvalid(shape)

    if valid:
        a[idx] # doesn't raise
    else:
        raises(IndexError, lambda: a[idx])
