from pytest import raises

from numpy import array, arange, isin, prod, unique

from hypothesis import given, assume, example
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from ..integerarray import IntegerArray
from ..tuple import Tuple
from .helpers import ndindices, shapes, assert_equal

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
@given(ndindices, ndindices, one_of(integers(0, 100), shapes))
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
    except DeprecationWarning as w:
        if "Out of bound index found. This was previously ignored when the indexing result contained no elements. In the future the index error will be raised. This error occurs either due to an empty slice, or if an array has zero elements even before indexing." in w.args[0]:
            assume(False)
        else:
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

        if (isinstance(index2, IntegerArray)
            or (isinstance(index2, Tuple)
                and any(isinstance(i, IntegerArray) for i in index2.args))):
            # isin(x, y) has the same shape as x. If idx2 has an integer array
            # it may index the same element more than once, but idx1 will not.
            assert_equal(unique(asubindex.flatten()), unique(a1[isin(a1, a2)]))
        else:
            assert_equal(asubindex.flatten(), a1[isin(a1, a2)])

        try:
            subindex2 = index2.as_subindex(index1)
        except NotImplementedError:
            return
        asubindex2 = a1[subindex2.raw]
        assert_equal(asubindex2, asubindex)
