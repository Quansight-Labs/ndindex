from itertools import product

from numpy import arange, isin

from hypothesis import given, assume, example
from hypothesis.strategies import integers, one_of

from pytest import raises

from ..ndindex import ndindex
from ..tuple import Tuple
from ..integer import Integer
from .helpers import (check_same, Tuples, prod, shapes, iterslice, ndindices,
                      slices, assert_equal)


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

@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_reduce_no_shape_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        idx = Tuple(*t)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    check_same(a, idx.raw, func=lambda x: x.reduce(),
               same_exception=False)

    reduced = idx.reduce()
    if isinstance(reduced, Tuple):
        assert len(reduced.args) != 1
        assert reduced == () or reduced.args[-1] != ...

@example((0, 1, ..., 2, 3), (2, 3, 4, 5, 6, 7))
@example((0, slice(None), ..., slice(None), 3), (2, 3, 4, 5, 6, 7))
@example((0, ..., slice(None)), (2, 3, 4, 5, 6, 7))
@example((slice(None, None, -1),), (2,))
@example((..., slice(None, None, -1),), (2, 3, 4))
@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_reduce_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        index = Tuple(*t)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    check_same(a, index.raw, func=lambda x: x.reduce(shape),
               same_exception=False)

    try:
        reduced = index.reduce(shape)
    except IndexError:
        pass
    else:
        if isinstance(reduced, Tuple):
            assert len(reduced.args) != 1
            assert reduced == () or reduced.args[-1] != ...
        # TODO: Check the other properties from the Tuple.reduce docstring.

def test_tuple_reduce_explicit():
    # Some aspects of Tuple.reduce are hard to test as properties, so include
    # some explicit tests here.

    # (Before Index, shape): After index
    tests = {
        # Make sure redundant slices are removed
        (Tuple(0, ..., slice(0, 3)), (5, 3)): Integer(0),
        (Tuple(slice(0, 5), ..., 0), (5, 3)): Tuple(..., Integer(0)),
        # Ellipsis is removed if unnecessary
        (Tuple(0, ...), (2, 3)): Integer(0),
        (Tuple(0, 1, ...), (2, 3)): Tuple(Integer(0), Integer(1)),
        (Tuple(..., 0, 1), (2, 3)): Tuple(Integer(0), Integer(1)),
    }

    for (before, shape), after in tests.items():
        reduced = before.reduce(shape)
        assert reduced == after

        a = arange(prod(shape)).reshape(shape)
        check_same(a, before.raw, func=lambda x: x.reduce(shape))

@example((0, 1, ..., 2, 3), (2, 3, 4, 5, 6, 7))
@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_expand_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        index = Tuple(*t)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    check_same(a, index.raw, func=lambda x: x.expand(shape),
               same_exception=False)

    try:
        expanded = index.expand(shape)
    except IndexError:
        pass
    else:
        assert isinstance(expanded, Tuple)
        assert ... not in expanded.args
        if isinstance(shape, int):
            assert len(expanded.args) == 1
        else:
            assert len(expanded.args) == len(shape)

# This is here because expand() always returns a Tuple, so it is very similar
# to the test_tuple_expand_hypothesis test.
@given(ndindices(), one_of(shapes, integers(0, 10)))
def test_ndindex_expand_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = ndindex(idx)

    check_same(a, index.raw, func=lambda x: x.expand(shape),
               same_exception=False)


    try:
        expanded = index.expand(shape)
    except IndexError:
        pass
    else:
        assert isinstance(expanded, Tuple)
        assert ... not in expanded.args
        if isinstance(shape, int):
            assert len(expanded.args) == 1
        else:
            assert len(expanded.args) == len(shape)

@example((0, slice(None), ..., slice(None), 3), (2, 3, 4, 5, 6, 7))
@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_newshape_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        index = Tuple(*t)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    # Call newshape so we can see if any exceptions match
    def func(t):
        t.newshape(shape)
        return t

    def assert_equal(x, y):
        newshape = index.newshape(shape)
        assert x.shape == y.shape == newshape

    check_same(a, index.raw, func=func, assert_equal=assert_equal,
               same_exception=False)

def test_tuple_newshape_ndindex_input():
    raises(TypeError, lambda: Tuple(1).newshape(Tuple(2, 1)))
    raises(TypeError, lambda: Tuple(1).newshape(Integer(2)))


@example((0,), (slice(1, 2),), 3)
@given(Tuples, slices(), one_of(shapes, integers(0, 10)))
def test_tuple_as_subindex_slice_hypothesis(t, index, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        T = Tuple(*t)
        Index = ndindex(index)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    empty = False
    try:
        Subindex = T.as_subindex(Index)
    except NotImplementedError: # pragma: no cover
        return
    except ValueError as e:
        assert "do not intersect" in e.args[0]
        empty = True

    try:
        aT = a[t]
        aindex = a[index]
    except IndexError: # pragma: no cover
        assume(False)

    if empty:
        assert not isin(aT, aindex).any()
        assert not isin(aindex, aT).any()
    else:
        asubindex = aindex[Subindex.raw]

        # TODO: how can we check that the shape is correct?
        assert_equal(asubindex.flatten(), aT[isin(aT, aindex)])

@example((), (slice(None, None, -1),), (2,))
@example((), (..., slice(None, None, -1),), (2,))
@example((slice(0, 1),), (2,), (3,))
@example((slice(0, 5), slice(0, 5)), (slice(3, 10), slice(3, 10)), (20, 20))
@given(Tuples, Tuples, shapes)
def test_tuple_as_subindex_tuple_hypothesis(t, index, shape):
    a = arange(prod(shape)).reshape(shape)

    try:
        T = Tuple(*t)
        Index = ndindex(index)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    empty = False
    try:
        Subindex = T.as_subindex(Index)
    except NotImplementedError: # pragma: no cover
        return
    except ValueError as e:
        assert "do not intersect" in e.args[0]
        empty = True


    try:
        aT = a[t]
        aindex = a[index]
    except IndexError: # pragma: no cover
        assume(False)

    if empty:
        assert not isin(aT, aindex).any()
        assert not isin(aindex, aT).any()
    else:
        asubindex = aindex[Subindex.raw]

        # TODO: how can we check that the shape is correct?
        assert_equal(asubindex.flatten(), aT[isin(aT, aindex)])

@example((0, slice(0, 0)), (1, 2))
@given(Tuples, one_of(shapes, integers(0, 10)))
def test_tuple_isempty_hypothesis(t, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        T = Tuple(*t)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    # Call isempty to see if the exceptions are the same
    def func(T):
        T.isempty(shape)
        return T

    def assert_equal(a_raw, a_idx):
        isempty = T.isempty()

        aempty = (a_raw.size == 0)
        assert aempty == (a_idx.size == 0)

        # If isempty is True then a[t] should be empty
        if isempty:
            assert aempty, (T, shape)
        # We cannot test the converse with hypothesis. isempty may be False
        # but a[t] could still be empty for this specific a (e.g., if a is
        # already itself empty).

        # isempty() should always give the correct result for a specific
        # array after reduction
        assert T.isempty(shape) == aempty, (T, shape)

    check_same(a, t, func=func, assert_equal=assert_equal,
               same_exception=False)
