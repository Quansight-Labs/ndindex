from pytest import raises

from numpy import array, arange, asarray, bincount, isin, intp

from hypothesis import given, assume, example

from ..ndindex import ndindex
from ..integerarray import IntegerArray
from ..slice import Slice
from ..tuple import Tuple
from .helpers import (as_subindex_indices1, as_subindex_indices2, prod,
                      short_shapes, assert_equal, warnings_are_errors)

@example((slice(0, 8), slice(0, 9), slice(0, 10)),
         ([2, 5, 6, 7], slice(1, 9, 1), slice(5, 10, 1)),
         (20, 20, 20))
@example((), (None, array([], dtype=intp)), 0)
@example((), array([], dtype=bool), 0)
@example((), IntegerArray(0), 1)
@example((), (None, True), 0)
@example(..., False, 0)
@example(..., (False,), 0)
@example((IntegerArray(2),), (slice(0, 1),), 3)
@example(([0],), (slice(1, 2),), 3)
@example(array([True]), slice(0, 1), 2)
@example((slice(0, 2), slice(0, 2)), (array([0, 2]), array([0, 2])), 3)
@example(IntegerArray([0, 1]), (slice(0, 1),), 2)
@example(array([1]), slice(0, 1), 1)
@example(IntegerArray(0), slice(0, 1), 3)
@example(IntegerArray(0), slice(1, 2), 3)
@example(IntegerArray(2), slice(0, 1), 3)
@example(..., (-1, array([0])), (1, 1))
@example(..., (-1, array([0, 0])), (1, 1))
@example((slice(None, 1, None), slice(None, 1, None)),
         (array(0), array([0, 0])),
         (1, 1))
@example([[0, 11], [0, 0]], slice(0, 10), 20)
@example(slice(0, 0), 9007199254741193, 1)
@example((0,), (slice(1, 2),), 3)
@example(slice(0, 10), slice(5, 15), 20)
@example((), (slice(None, None, -1),), (2,))
@example((), (..., slice(None, None, -1),), (2,))
@example((slice(0, 1),), (2,), (3,))
@example((slice(0, 5), slice(0, 5)), (slice(3, 10), slice(3, 10)), (20, 20))
@example((slice(0, 5), slice(0, 5)), (1, 1), (10, 10))
@example(0, slice(0, 0), 1)
@example([0], slice(0, 0), 1)
@example(0, slice(0, 1), 1)
@example([0], slice(0, 1), 1)
@example(slice(0, 5), 2, 10)
@example(0, (slice(None, 0, None), Ellipsis), 1)
@example(0, (slice(1, 2),), 1)
@example((slice(0, 1), slice(0, 1)), (array([0, 1]), array([1, 0])), (2, 3))
@example((array([0, 1]), array([1, 0])), (slice(0, 1), slice(0, 1)), (2, 3))
@example(slice(None, 5), array([[0]]), 1)
@example([[0, 1], [0, 0]], slice(0, 10), 20)
@example((slice(None, 1), array([[[0, 0], [0, 0]]]), slice(None)),
         array([[[0, 0], [0, 0]]]),
         (1, 2, 2))
@example((slice(0, 2), array([0, 1])), (array([0, 1, 2]),), (3, 3))
@example((array([0, 1]), slice(0, 2)), (slice(0, 2), array([0, 1, 2])), (3, 3))
@example((slice(None, 2), array([[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]), slice(None)),
         array([True, True, True]),
         (3, 1, 3))
@given(as_subindex_indices1, as_subindex_indices2, short_shapes)
@warnings_are_errors
def test_as_subindex_hypothesis(idx1, idx2, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    try:
        index1 = ndindex(idx1)
        index2 = ndindex(idx2)
    except ValueError: # pragma: no cover
        assume(False)

    empty = False
    try:
        Subindex = index1.as_subindex(index2)
    except NotImplementedError:
        return
    except ValueError as e:
        assert "do not intersect" in e.args[0]
        empty = True

    try:
        a1 = a[index1.raw]
        a2 = a[index2.raw]
    except IndexError: # pragma: no cover
        assume(False)
    except DeprecationWarning as w: # pragma: no cover
        if "Out of bound index found. This was previously ignored when the indexing result contained no elements. In the future the index error will be raised. This error occurs either due to an empty slice, or if an array has zero elements even before indexing." in w.args[0]:
            assume(False)
        else: # pragma: no cover
            raise

    if empty:
        assert not isin(a1, a2).any()
        assert not isin(a2, a1).any()
        with raises(ValueError, match="do not intersect"):
            try:
                index2.as_subindex(index1)
            except NotImplementedError:
                raise ValueError('do not intersect')
    else:
        asubindex = a2[Subindex.raw]

        # asubindex should have exactly the elements of a2 that are also in
        # a1. An index with an integer array can index the same element more
        # than once, in which case asubindex has that element once for each
        # time it is in a1 and for each time it is in a2.
        def counts(x):
            return bincount(asarray(x).flatten(), minlength=a.size)
        assert_equal(counts(asubindex), counts(a1)*counts(a2))

        index2_args = index2.args if isinstance(index2, Tuple) else (index2,)
        if not any(isinstance(i, IntegerArray) for i in index2_args):
            # isin(x, y) has the same shape as x. If idx2 does not have an
            # integer array, the elements are in the same order as in a1.
            assert_equal(asubindex.flatten(), a1[isin(a1, a2)])

        if (all(isinstance(i, Slice) for i in index2_args)
            and a1.size != 0 and isin(a1, a2).all()):
            # If idx2 only has slices, a2 has the same dimensions as a. Then
            # if a1 is a subarray of a2, asubindex is exactly a1, including
            # the shape. (a[()] is a scalar but a[...] is a 0-D array.)
            assert_equal(asubindex, a1, allow_scalar_0d=True)

        try:
            subindex2 = index2.as_subindex(index1)
        except NotImplementedError:
            return
        asubindex2 = a1[subindex2.raw]
        assert_equal(asubindex2, asubindex)
