from numpy import arange, prod

from hypothesis import given, example
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from ..tuple import Tuple
from .helpers import ndindices, shapes, check_same


@example(([0, 1], 0), (2, 2))
@example((..., [0, 1], 0), (2, 2))
@example((..., None, 0), 1)
@example((0, 1, ..., 2, 3), (2, 3, 4, 5, 6, 7))
@example(None, 2)
@given(ndindices, one_of(shapes, integers(0, 10)))
def test_expand_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = ndindex(idx)

    try:
        expanded = index.expand(shape)
    except IndexError:
        pass
    except NotImplementedError:
        return
    else:
        assert isinstance(expanded, Tuple)
        assert ... not in expanded.args
        if isinstance(idx, tuple):
            n_newaxis = index.args.count(None)
        elif index == None:
            n_newaxis = 1
        else:
            n_newaxis = 0
        if isinstance(shape, int):
            assert len(expanded.args) == 1 + n_newaxis
        else:
            assert len(expanded.args) == len(shape) + n_newaxis

    check_same(a, index.raw, ndindex_func=lambda a, x: a[x.expand(shape).raw],
               same_exception=False)
