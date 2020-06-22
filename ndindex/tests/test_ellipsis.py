from numpy import arange, isin
from numpy.testing import assert_equal

from hypothesis import given, assume

from ..ndindex import ndindex
from ..slice import Slice
from ..tuple import Tuple
from .helpers import (check_same, prod, shapes, ellipses, positive_slices, Tuples)

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

@given(ellipses(), positive_slices, shapes)
def test_ellipsis_as_subindex_slice_hypothesis(idx, index, shape):
    a = arange(prod(shape)).reshape(shape)

    E = ndindex(idx)
    try:
        Index = Slice(index)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    try:
        Subindex = E.as_subindex(Index)
    except NotImplementedError: # pragma: no cover
        return

    try:
        aE = a[idx]
        aindex = a[index]
    except IndexError: # pragma: no cover
        assume(False)
    asubindex = aindex[Subindex.raw]

    assert_equal(asubindex.flatten(), aE[isin(aE, aindex)])

@given(ellipses(), Tuples, shapes)
def test_ellipsis_as_subindex_tuple_hypothesis(idx, index, shape):
    a = arange(prod(shape)).reshape(shape)

    E = ndindex(idx)
    try:
        Index = Tuple(*index)
    except (IndexError, ValueError): # pragma: no cover
        assume(False)

    try:
        Subindex = E.as_subindex(Index)
    except NotImplementedError: # pragma: no cover
        return

    try:
        aE = a[idx]
        aindex = a[index]
    except IndexError: # pragma: no cover
        assume(False)
    asubindex = aindex[Subindex.raw]

    assert_equal(asubindex.flatten(), aE[isin(aE, aindex)])
