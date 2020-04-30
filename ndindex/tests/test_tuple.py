from itertools import product

from numpy import arange

from hypothesis import given, assume, example
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from ..tuple import Tuple
from .helpers import check_same, Tuples, prod, shapes, iterslice


def test_tuple_exhaustive():
    # Exhaustive tests here have to be very limited because of combinatorial
    # explosion.
    a = arange(2*2*2).reshape((2, 2, 2))
    types = {
        slice: lambda: iterslice((-1, 1), (-1, 1), (-1, 1), one_two_args=False),
        # slice: _iterslice,
        int: lambda: ((i,) for i in range(-3, 3)),
        type(...): lambda: ()
    }

    for t1, t2, t3 in product(types, repeat=3):
        for t1_args in types[t1]():
            for t2_args in types[t2]():
                for t3_args in types[t3]():
                    idx1 = t1(*t1_args)
                    idx2 = t2(*t2_args)
                    idx3 = t3(*t3_args)

                    index = (idx1, idx2, idx3)
                    # Disable the same exception check because there could be
                    # multiple invalid indices in the tuple, and for instance
                    # numpy may give an IndexError but we would give a
                    # TypeError because we check the type first.
                    check_same(a, index, same_exception=False)
                    try:
                        idx = Tuple(*index)
                    except (IndexError, ValueError):
                        pass
                    else:
                        assert idx.has_ellipsis == (type(...) in (t1, t2, t3))

@given(Tuples, shapes)
def test_tuples_hypothesis(t, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, t, same_exception=False)

@given(Tuples, shapes)
def test_ellipsis_index(t, shape):
    a = arange(prod(shape)).reshape(shape)
    try:
        idx = ndindex(t)
    except (IndexError, ValueError):
        pass
    else:
        if isinstance(idx, Tuple):
            # Don't know if there is a better way to test ellipsis_idx
            check_same(a, t, func=lambda x: ndindex((*x.raw[:x.ellipsis_index], ..., *x.raw[x.ellipsis_index+1:])))

@example((0, 1, ..., 2, 3), (5, 5, 5, 5, 5, 5))
@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_reduce_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        idx = Tuple(*t)
    except (IndexError, ValueError):
        assume(False)

    check_same(a, idx.raw, func=lambda x: x.reduce(shape),
               same_exception=False)

@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_reduce_no_shape_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        idx = Tuple(*t)
    except (IndexError, ValueError):
        assume(False)

    check_same(a, idx.raw, func=lambda x: x.reduce(),
               same_exception=False)
