from itertools import chain, product

from numpy import arange

from .helpers import check_same
from ..ndindex import Slice, Integer, Tuple

def _iterslice(start_range=(-10, 10), stop_range=(-10, 10), step_range=(-10, 10)):
    for start in chain(range(*start_range), [None]):
        for stop in chain(range(*stop_range), [None]):
            for step in chain(range(*step_range), [None]):
                yield (start, stop, step)

def test_slice():
    a = arange(100)
    for start, stop, step in _iterslice():
        raw_type = slice
        raw_args = start, stop, step
        idx_type = Slice
        idx_args = start, stop, step
        check_same(a, raw_type, raw_args, idx_type, idx_args)

def test_integer():
    a = arange(10)
    for i in range(-12, 12):
        raw_type = int
        raw_args = (i,)
        idx_type = Integer
        idx_args = (i,)
        check_same(a, raw_type, raw_args, idx_type, idx_args)

def test_tuple():
    # Exhaustive tests here have to be very limited because of combinatorial
    # explosion.
    a = arange(2*2*2).reshape((2, 2, 2))
    types = {
        slice: lambda: _iterslice((-1, 1), (-1, 1), (-1, 1)),
        # slice: _iterslice,
        int: lambda: ((i,) for i in range(-3, 3)),
    }

    for t1, t2, t3 in product(types, repeat=3):
        print(t1, t2, t3)
        for t1_args in types[t1]():
            for t2_args in types[t2]():
                for t3_args in types[t3]():
                    idx1 = t1(*t1_args)
                    idx2 = t2(*t2_args)
                    idx3 = t3(*t3_args)

                    raw_type = lambda *x: tuple(x)
                    raw_args = idx1, idx2, idx3
                    idx_type = Tuple
                    idx_args = raw_args
                    # Disable the same exception check because there could be
                    # multiple invalid indices in the tuple, and for instance
                    # numpy may give an IndexError but we would give a
                    # TypeError because we check the type first.
                    check_same(a, raw_type, raw_args, idx_type, idx_args, same_exception=False)
