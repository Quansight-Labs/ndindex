from numpy import arange

from .helpers import raises_same, check_same
from ..ndindex import Slice, Integer

def test_slice():
    a = arange(100)
    for start in range(-10, 10):
        for stop in range(-10, 10):
            for step in range(-10, 10):
                raw_type = slice
                raw_args = start, stop, step
                idx_type = Slice
                idx_args = start, stop, step
                check_same(a, raw_type, raw_args, idx_type, idx_args,
                           raises=step == 0)

def test_integer():
    a = arange(10)
    for i in range(-12, 12):
        raw_type = int
        raw_args = (i,)
        idx_type = Integer
        idx_args = (i,)
        check_same(a, raw_type, raw_args, idx_type, idx_args,
                   raises=(i < -10 or i >= 10))
