"""
Tests are extremely important for ndindex. All operations should produce
correct results. We test this by checking against numpy arange (the array
values do not matter, so long as they are distinct).

There are two primary types of tests that we employ to verify this:

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

Why bother with hypothesis if the same thing is already tested exhaustively?
The main reason is that hypothesis is much better at producing human-readable
failure examples. When an exhaustive test fails, the failure will always be
from the first set of inputs in the loop that produces a failure. Hypothesis
on the other hand attempts to "shrink" the failure input to smallest input
that still fails. For example, a failing exhaustive slice test might give
Slice(-10, -9, -10) as a the failing example, but hypothesis would shrink it
to Slice(-2, -1, -1). Another reason for the duplication is that hypothesis
can sometimes test a slightly expanded test space without any additional
consequences. For example, test_slice_reduce_hypothesis() tests all types of
array shapes, whereas test_slice_reduce_exhaustive() tests only 1-dimensional
shapes. This doesn't affect things because hypotheses will always shrink large
shapes to a 1-dimensional shape in the case of a failure. Consequently every
exhaustive test should have a corresponding hypothesis test.

"""

from itertools import chain, product

from pytest import raises

from numpy import arange

from hypothesis import given, assume
from hypothesis.strategies import integers, one_of

from ..ndindex import Slice, Integer, Tuple, ndindex
from .helpers import (check_same, ints, slices, Tuples, prod, shapes,
                      ndindices, ellipses)

# Variable naming conventions in this file:

# a: numpy arange. May be reshaped to be multidimensional
# shape: a tuple of integers
# i: integer used as an integer index
# idx: ndindex type
# s: slice (Python type)
# S: Slice (ndindex type)
# size: integer passed to arange

def _iterslice(start_range=(-10, 10),
               stop_range=(-10, 10),
               step_range=(-10, 10),
               one_two_args=True
):
    # one_two_args is unnecessary if the args are being passed to slice(),
    # since slice() already canonicalizes missing arguments to None. We do it
    # for Slice to test that behavior.
    if one_two_args:
        for start in chain(range(*start_range), [None]):
            yield (start,)

        for start in chain(range(*start_range), [None]):
            for stop in chain(range(*stop_range), [None]):
                yield (start, stop)

    for start in chain(range(*start_range), [None]):
        for stop in chain(range(*stop_range), [None]):
            for step in chain(range(*step_range), [None]):
                yield (start, stop, step)

def test_slice_args():
    # Test the behavior when not all three arguments are given
    # TODO: Incorporate this into the normal slice tests
    raises(TypeError, lambda: slice())
    raises(TypeError, lambda: Slice())

    S = Slice(1)
    assert S == Slice(S) == Slice(None, 1) == Slice(None, 1, None) == Slice(None, 1, None)
    assert S.raw == slice(None, 1, None)
    assert S.args == (S.start, S.stop, S.step)

    S = Slice(0, 1)
    assert S == Slice(S) == Slice(0, 1, None)
    assert S.raw == slice(0, 1, None)
    assert S.args == (S.start, S.stop, S.step)

    S = Slice(0, 1, 2)
    assert S == Slice(S)
    assert S.raw == slice(0, 1, 2)
    assert S.args == (S.start, S.stop, S.step)

def test_slice_exhaustive():
    for n in range(100):
        a = arange(n)
        for start, stop, step in _iterslice(one_two_args=False):
            check_same(a, slice(start, stop, step))

@given(slices(), integers(0, 100))
def test_slice_hypothesis(s, size):
    a = arange(size)
    check_same(a, s)

def test_slice_len_exhaustive():
    for args in _iterslice():
        try:
            S = Slice(*args)
        except ValueError:
            continue
        try:
            l = len(S)
        except ValueError:
            # No maximum
            l = 10000

        m = -1
        for n in range(20):
            a = arange(n)
            L = len(a[S.raw])
            assert L <= l, S
            m = max(L, m)
        if l != 10000:
            assert m == l, S
        else:
            # If there is no maximum, the size of the slice should increase
            # with larger arrays.
            assert len(arange(30)[S.raw]) > m, S

        # TODO
        # if l == 0:
        #     # There should only be one canonical length 0 slice
        #     assert s == Slice(0, 0)

@given(slices())
def test_slice_len_hypothesis(s):
    try:
        S = Slice(s)
    except ValueError:
        assume(False)
    try:
        l = len(S)
    except ValueError:
        # No maximum
        l = 10000

    m = -1
    for n in range(20):
        a = arange(n)
        L = len(a[S.raw])
        assert L <= l, (S, n)
        m = max(L, m)
    if l != 10000:
        assert m == l, S
    else:
        # If there is no maximum, the size of the slice should increase
        # with larger arrays.
        assert len(arange(30)[S.raw]) > m, S

def test_slice_args_reduce_no_shape():
    S = Slice(1).reduce()
    assert S == Slice(None, 1).reduce() == Slice(0, 1, None).reduce() == Slice(0, 1).reduce() == Slice(0, 1, 1)

    S = Slice(0, 1).reduce()
    assert S == Slice(0, 1, None).reduce() == Slice(0, 1, 1)

def test_slice_reduce_no_shape_exhaustive():
    for n in range(10):
        a = arange(n)
        for args in _iterslice():
            try:
                S = Slice(*args)
            except ValueError:
                continue

            check_same(a, S.raw, func=lambda x: x.reduce())

            # Check the conditions stated by the Slice.reduce() docstring
            reduced = S.reduce()
            # TODO: Test that start and stop are not None when possible
            assert reduced.step != None

@given(slices(), shapes)
def test_slice_reduce_no_shape_hypothesis(s, shape):
    a = arange(prod(shape)).reshape(shape)
    try:
        S = Slice(s)
    except ValueError:
        assume(False)

    # The axis argument is tested implicitly in the Tuple.reduce test. It is
    # difficult to test here because we would have to pass in a Tuple to
    # check_same.
    check_same(a, S.raw, func=lambda x: x.reduce())

    # Check the conditions stated by the Slice.reduce() docstring
    reduced = S.reduce()
    # TODO: Test that start and stop are not None when possible
    assert reduced.step != None

def test_slice_reduce_exhaustive():
    for n in range(10):
        a = arange(n)
        for args in _iterslice():
            try:
                S = Slice(*args)
            except ValueError:
                continue

            check_same(a, S.raw, func=lambda x: x.reduce((n,)))

            # Check the conditions stated by the Slice.reduce() docstring
            reduced = S.reduce((n,))
            assert reduced.start >= 0
            # We cannot require stop > 0 because if stop = None and step < 0, the
            # only equivalent stop that includes 0 is negative.
            assert reduced.stop != None
            assert reduced.step != None
            assert len(reduced) == len(a[reduced.raw]), (S, n)

@given(slices(), shapes)
def test_slice_reduce_hypothesis(s, shape):
    a = arange(prod(shape)).reshape(shape)
    try:
        S = Slice(s)
    except ValueError:
        assume(False)

    # The axis argument is tested implicitly in the Tuple.reduce test. It is
    # difficult to test here because we would have to pass in a Tuple to
    # check_same.
    check_same(a, S.raw, func=lambda x: x.reduce(shape))

    # Check the conditions stated by the Slice.reduce() docstring
    try:
        reduced = S.reduce(shape)
    except IndexError:
        # shape == ()
        return

    assert reduced.start >= 0
    # We cannot require stop > 0 because if stop = None and step < 0, the
    # only equivalent stop that includes 0 is negative.
    assert reduced.stop != None
    assert reduced.step != None
    assert len(reduced) == len(a[reduced.raw]), (S, shape)

def test_integer_exhaustive():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i)

@given(ints(), integers(5, 100))
def test_integer_hypothesis(i, size):
    a = arange(size)
    check_same(a, i)

def test_integer_len_exhaustive():
    for i in range(-12, 12):
        idx = Integer(i)
        assert len(idx) == 1

@given(ints())
def test_integer_len_hypothesis(i):
    idx = Integer(i)
    assert len(idx) == 1

def test_integer_reduce_exhaustive():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i, func=lambda x: x.reduce((10,)))

        try:
            reduced = Integer(i).reduce(10)
        except IndexError:
            pass
        else:
            assert reduced.raw >= 0

@given(integers(0, 10), shapes)
def test_integer_reduce_hypothesis(i, shape):
    a = arange(prod(shape)).reshape(shape)
    # The axis argument is tested implicitly in the Tuple.reduce test. It is
    # difficult to test here because we would have to pass in a Tuple to
    # check_same.
    check_same(a, i, func=lambda x: x.reduce(shape))

    try:
        reduced = Integer(i).reduce(shape)
    except IndexError:
        pass
    else:
        assert reduced.raw >= 0

def test_integer_reduce_no_shape_exhaustive():
    a = arange(10)
    for i in range(-12, 12):
        check_same(a, i, func=lambda x: x.reduce())

@given(integers(0, 10), shapes)
def test_integer_reduce_no_shape_hypothesis(i, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, i, func=lambda x: x.reduce())

def test_ellipsis_exhaustive():
    for n in range(10):
        a = arange(n)
    check_same(a, ...)

@given(ellipses(), shapes)
def test_ellipsis_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx)

def test_ellipsis_reduce_exhaustive():
    for n in range(10):
        a = arange(n)
        check_same(a, ..., func=lambda x: x.reduce((n,)))

@given(ellipses(), shapes)
def test_ellipsis_reduce_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, func=lambda x: x.reduce(shape))

def test_ellipsis_reduce_no_shape_exhaustive():
    for n in range(10):
        a = arange(n)
        check_same(a, ..., func=lambda x: x.reduce())

@given(ellipses(), shapes)
def test_ellipsis_reduce_no_shape_hypothesis(idx, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, idx, func=lambda x: x.reduce())

def test_tuple_exhaustive():
    # Exhaustive tests here have to be very limited because of combinatorial
    # explosion.
    a = arange(2*2*2).reshape((2, 2, 2))
    types = {
        slice: lambda: _iterslice((-1, 1), (-1, 1), (-1, 1), one_two_args=False),
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

@given(ndindices())
def test_eq(idx):
    new = type(idx)(*idx.args)
    assert (new == idx) is True
    assert (new.raw == idx.raw) is True
    assert hash(new) == hash(idx)
    assert (idx == idx.raw) is True
    assert (idx.raw == idx) is True
    assert (idx == 'a') is False
    assert ('a' == idx) is False
    assert (idx != 'a') is True
    assert ('a' != idx) is True

@given(ndindices())
def test_ndindex(idx):
    assert ndindex(idx) == idx
    assert ndindex(idx).raw == idx
    ix = ndindex(idx)
    assert ndindex(ix.raw) == ix
