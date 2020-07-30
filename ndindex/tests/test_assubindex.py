from pytest import raises

from numpy import arange, isin, prod

from hypothesis import given, assume, example
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from .helpers import ndindices, shapes, assert_equal

@example(slice(0, 0), 9007199254741193, 1)
@example((0,), (slice(1, 2),), 3)
@example(slice(0, 10), slice(5, 15), 20)
@example((), (slice(None, None, -1),), (2,))
@example((), (..., slice(None, None, -1),), (2,))
@example((slice(0, 1),), (2,), (3,))
@example((slice(0, 5), slice(0, 5)), (slice(3, 10), slice(3, 10)), (20, 20))
@example((slice(0, 5), slice(0, 5)), (1, 1), (10, 10))
@example(0, slice(0, 0), 1)
@example(0, slice(0, 1), 1)
@example(slice(0, 5), 2, 10)
@example(0, (slice(None, 0, None), Ellipsis), 1)
@example(0, (slice(1, 2),), 1)
@given(ndindices(), ndindices(), one_of(integers(0, 100), shapes))
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
        a1 = a[idx1]
        a2 = a[idx2]
    except IndexError: # pragma: no cover
        assume(False)

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

        assert_equal(asubindex.flatten(), a1[isin(a1, a2)])

        try:
            subindex2 = index2.as_subindex(index1)
        except NotImplementedError:
            return
        asubindex2 = a1[subindex2.raw]
        assert_equal(asubindex2, asubindex)
