from pytest import raises

from numpy import arange

from hypothesis import given
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from ..tuple import Tuple
from ..integer import Integer
from .helpers import ndindices, check_same, short_shapes, prod

@given(ndindices, one_of(short_shapes, integers(0, 10)))
def test_selected_indices_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        ndindex(idx)
    except IndexError:
        pass

    def raw_func(a, idx):
        return list(a[idx].flat)

    def ndindex_func(a, index):
        indices = list(index.selected_indices(shape))
        for i in indices:
            if len(a.shape) == 1:
                assert isinstance(i, Integer)
            else:
                assert isinstance(i, Tuple)
                assert all(isinstance(j, Integer) for j in i.args)

        return [a[i.raw] for i in indices]

    def assert_equal(a, b):
        assert a == b

    check_same(a, idx, raw_func=raw_func, ndindex_func=ndindex_func,
               assert_equal=assert_equal)
