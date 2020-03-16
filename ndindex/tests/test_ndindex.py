"""
Tests are extremely important for ndindex. All operations should produce
correct results. We test this by checking against numpy arange (the array
values do not matter, so long as they are distinct).

There are two primary types of tests that we employ to verify this

- Exhaustive tests. These test every possible value in some range. See for
  example test_slice. This is the best type of test, but unfortunately, it is
  often impossible to do due to combinatorial explosion.

- Hypothesis tests. Hypothesis is a library that can intelligently check a
  combinatorial search space. This requires writing hypothesis strategies that
  can generate all the relevant types of indices (see helpers.py). For more
  information on hypothesis, see
  https://hypothesis.readthedocs.io/en/latest/index.html.

The basic idea in both cases is the same. Take the pure index and the
ndindex(index).raw, or in the case of a transformation, the before and after
raw index, and index an arange with them. If they do not give the same output
array, or do not both produce the same error, the code is not correct.

"""

from itertools import chain, product

from numpy import arange

from hypothesis import given, assume
from hypothesis.strategies import integers, one_of

from ..ndindex import Slice
from .helpers import check_same, ints, slices, tuples, prod, shapes

def _iterslice(start_range=(-10, 10), stop_range=(-10, 10), step_range=(-10, 10)):
    for start in chain(range(*start_range), [None]):
        for stop in chain(range(*stop_range), [None]):
            for step in chain(range(*step_range), [None]):
                yield (start, stop, step)

def test_slice():
    a = arange(100)
    for start, stop, step in _iterslice():
        check_same(a, slice(start, stop, step))

@given(slices(), integers(5, 100))
def test_slice_hypothesis(s, size):
    a = arange(size)
    check_same(a, s)

def test_slice_len():
    for start, stop, step in _iterslice():
        try:
            s = Slice(start, stop, step)
        except ValueError:
            continue
        try:
            l = len(s)
        except ValueError:
            # No maximum
            l = 10000

        m = -1
        for n in range(20):
            a = arange(n)
            L = len(a[s.raw])
            assert L <= l, s
            m = max(L, m)
        if l != 10000:
            assert m == l, s
        else:
            # If there is no maximum, the size of the slice should increase
            # with larger arrays.
            assert len(arange(30)[s.raw]) > m, s

@given(slices())
def test_slice_len_hypothesis(s):
    try:
        s = Slice(s)
    except ValueError:
        assume(False)
    try:
        l = len(s)
    except ValueError:
        # No maximum
        l = 10000

    m = -1
    for n in range(20):
        a = arange(n)
        L = len(a[s.raw])
        assert L <= l, (s, n)
        m = max(L, m)
    if l != 10000:
        assert m == l, s
    else:
        # If there is no maximum, the size of the slice should increase
        # with larger arrays.
        assert len(arange(30)[s.raw]) > m, s

def test_integer():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i)

@given(ints(), integers(5, 100))
def test_integer_hypothesis(idx, size):
    a = arange(size)
    check_same(a, idx)

def test_integer_reduce():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i, func=lambda x: x.reduce((10,)))

@given(integers(0, 10), shapes)
def test_integer_reduce_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    # The axis argument is tested implicitly in the Tuple.reduce test. It is
    # difficult to test here because we would have to pass in a Tuple to
    # check_same.
    check_same(a, idx, func=lambda x: x.reduce(shape))

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
        for t1_args in types[t1]():
            for t2_args in types[t2]():
                for t3_args in types[t3]():
                    idx1 = t1(*t1_args)
                    idx2 = t2(*t2_args)
                    idx3 = t3(*t3_args)

                    index = idx1, idx2, idx3
                    # Disable the same exception check because there could be
                    # multiple invalid indices in the tuple, and for instance
                    # numpy may give an IndexError but we would give a
                    # TypeError because we check the type first.
                    check_same(a, index, same_exception=False)

@given(tuples(one_of(ints(), slices())), shapes)
def test_tuples_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, same_exception=False)
