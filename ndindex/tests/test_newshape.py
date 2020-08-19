from pytest import raises

from numpy import arange, prod, array, full

from hypothesis import given, example
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from ..tuple import Tuple
from ..integer import Integer
from ..booleanarray import BooleanArray
from .helpers import ndindices, short_shapes, check_same

@example(array([[[True], [False]]]), (1, 1, 2))
@example(full((1, 9), False), (3, 3))
@example(([0, 1], 0), (2, 2))
@example(([0, 0, 0], [0, 0]), (2, 2))
@example((0, None, 0, ..., 0, None, 0), (2, 2, 2, 2, 2, 2, 2))
@example((0, slice(None), ..., slice(None), 3), (2, 3, 4, 5, 6, 7))
@given(ndindices, one_of(short_shapes, integers(0, 10)))
def test_newshape_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        index = ndindex(idx)
    except IndexError:
        pass
    else:
        if (isinstance(index, BooleanArray)
            and index.count_nonzero == 0
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

        # Make sure ndindex input gives an error
        raises(TypeError, lambda: index.newshape(Tuple(2, 1)))
        raises(TypeError, lambda: index.newshape(Integer(2)))

    def raw_func(a, idx):
        return a[idx].shape

    def ndindex_func(a, index):
        return index.newshape(shape)

    def assert_equal(raw_shape, newshape):
        assert raw_shape == newshape


    check_same(a, idx, raw_func=raw_func, ndindex_func=ndindex_func,
               assert_equal=assert_equal, same_exception=False)
