import inspect

import numpy as np

from hypothesis import given, example, settings
from hypothesis.strategies import integers

from pytest import raises, warns

from ..ndindex import ndindex, asshape, iter_indices, ncycles, BroadcastError, AxisError
from ..booleanarray import BooleanArray
from ..integer import Integer
from ..ellipsis import ellipsis
from ..integerarray import IntegerArray
from ..tuple import Tuple
from .helpers import (ndindices, check_same, assert_equal, prod,
                      mutually_broadcastable_shapes, skip_axes)

@given(ndindices)
def test_eq(idx):
    index = ndindex(idx)
    new = type(index)(*index.args)

    if isinstance(new.raw, np.ndarray):
        # trying to get a single value out of comparing two arrays requires all sorts of special handling, just let numpy do it
        assert np.array_equal(new.raw, index.raw)
    else:
        assert new.raw == index.raw

    assert (new == index)
    assert (new.raw == index)
    assert (new == index.raw)
    assert (index == new)
    assert (index.raw == new)
    assert (index == new.raw)

    assert (index.raw == index)
    assert hash(new) == hash(index)
    assert (index == index.raw)
    assert not (index == 'a')
    assert not ('a' == index)
    assert (index != 'a')
    assert ('a' != index)

    try:
        h = hash(idx)
    except TypeError:
        pass
    else:
        assert hash(index) == h

    try:
        h = hash(index.raw)
    except TypeError:
        pass
    else:
        assert hash(index) == h

def test_eq_array_raises():
    index = ndindex([1, 2, 3])
    with raises(TypeError):
        np.equal(index.raw, index)
    with raises(TypeError):
        np.array_equal(index.raw, index)

def test_eq_explicit():
    assert Integer(0) != False
    assert Integer(1) != True
    assert Integer(0) != IntegerArray(0)
    assert IntegerArray([0, 1]) != [False, True]
    assert IntegerArray([0, 1]) == [0, 1]
    assert BooleanArray([False, True]) != [0, 1]
    assert BooleanArray([False, True]) == [False, True]

@example((np.array([1, 2]), 0))
@example([1, 2, 3])
@given(ndindices)
def test_ndindex(idx):
    index = ndindex(idx)
    assert index == idx
    def test_raw_eq(idx, index):
        if isinstance(idx, np.ndarray):
            assert_equal(index.raw, idx)
        elif isinstance(idx, list):
            assert index.dtype in [np.intp, np.bool_]
            assert_equal(index.raw, np.asarray(idx, dtype=index.dtype))
        elif isinstance(idx, tuple):
            assert type(index.raw) == type(idx)
            assert len(index.raw) == len(idx)
            assert index.args == index.raw
            for i, j in zip(idx, index.args):
                test_raw_eq(i, j)
        else:
            assert index.raw == idx
    test_raw_eq(idx, index)
    assert ndindex(index.raw) == index

def test_ndindex_invalid():
    a = np.arange(10)
    for idx in [1.0, [1.0], np.array([1.0]), np.array([1], dtype=object),
                np.array([])]:
        check_same(a, idx)

    # This index is allowed by NumPy, but gives a deprecation warnings. We are
    # not going to allow indices that give deprecation warnings in ndindex.
    with warns(None) as r: # Make sure no warnings are emitted from ndindex()
        raises(IndexError, lambda: ndindex([1, []]))
    assert not r

def test_ndindex_ellipsis():
    raises(IndexError, lambda: ndindex(ellipsis))

def test_signature():
    sig = inspect.signature(Integer)
    assert sig.parameters.keys() == {'idx'}


@example(([0, 1],))
@example((IntegerArray([], (0, 1)),))
@example(IntegerArray([], (0, 1)))
@example((1, ..., slice(1, 2)))
# eval can sometimes be slower than the default deadline of 200ms for large
# array indices
@settings(deadline=None)
@given(ndindices)
def test_repr_str(idx):
    # The repr form should be re-creatable
    index = ndindex(idx)
    d = {}
    exec("from ndindex import *", d)
    assert eval(repr(index), d) == idx

    # Str may not be re-creatable. Just test that it doesn't give an exception.
    str(index)

def test_asshape():
    assert asshape(1) == (1,)
    assert asshape(np.int64(2)) == (2,)
    assert type(asshape(np.int64(2))[0]) == int
    assert asshape((1, 2)) == (1, 2)
    assert asshape([1, 2]) == (1, 2)
    assert asshape((np.int64(1), np.int64(2))) == (1, 2)
    assert type(asshape((np.int64(1), np.int64(2)))[0]) == int
    assert type(asshape((np.int64(1), np.int64(2)))[1]) == int

    raises(TypeError, lambda: asshape(1.0))
    raises(TypeError, lambda: asshape((1.0,)))
    raises(ValueError, lambda: asshape(-1))
    raises(ValueError, lambda: asshape((1, -1)))
    raises(TypeError, lambda: asshape(...))
    raises(TypeError, lambda: asshape(Integer(1)))
    raises(TypeError, lambda: asshape(Tuple(1, 2)))
    raises(TypeError, lambda: asshape((True,)))

@example([((1, 1), (1, 1)), (1, 1)], (0, 0))
@example([((), (0,)), (0,)], (0,))
@example([((1, 2), (2, 1)), (2, 2)], 1)
@given(mutually_broadcastable_shapes, skip_axes())
def test_iter_indices(broadcastable_shapes, skip_axes):
    shapes, broadcasted_shape = broadcastable_shapes

    if skip_axes is None:
        res = iter_indices(*shapes)
        broadcasted_res = iter_indices(np.broadcast_shapes(*shapes))
        skip_axes = ()
    else:
        res = iter_indices(*shapes, skip_axes=skip_axes)
        broadcasted_res = iter_indices(np.broadcast_shapes(*shapes),
                                       skip_axes=skip_axes)

    if isinstance(skip_axes, int):
        skip_axes = (skip_axes,)

    sizes = [prod(shape) for shape in shapes]
    ndim = len(broadcasted_shape)
    arrays = [np.arange(size).reshape(shape) for size, shape in zip(sizes, shapes)]
    broadcasted_arrays = np.broadcast_arrays(*arrays)

    # Use negative indices to index the skip axes since only shapes that have
    # the skip axis will include a slice.
    normalized_skip_axes = sorted(ndindex(i).reduce(ndim).args[0] - ndim for i in skip_axes)
    skip_shapes = [tuple(shape[i] for i in normalized_skip_axes if -i <= len(shape)) for shape in shapes]
    broadcasted_skip_shape = tuple(broadcasted_shape[i] for i in normalized_skip_axes)

    broadcasted_non_skip_shape = tuple(broadcasted_shape[i] for i in range(-1, -ndim-1, -1) if i not in normalized_skip_axes)
    nitems = prod(broadcasted_non_skip_shape)
    broadcasted_nitems = prod(broadcasted_shape)

    vals = []
    n = -1
    try:
        for n, (idxes, bidxes) in enumerate(zip(res, broadcasted_res)):
            assert len(idxes) == len(shapes)
            for idx, shape in zip(idxes, shapes):
                assert isinstance(idx, Tuple)
                assert len(idx.args) == len(shape)
                for i in range(-1, -len(idx.args)-1, -1):
                    if i in normalized_skip_axes and len(idx.args) >= -i:
                        assert idx.args[i] == slice(None)
                    else:
                        assert isinstance(idx.args[i], Integer)

            aidxes = tuple([a[idx.raw] for a, idx in zip(arrays, idxes)])
            a_broadcasted_idxs = [a[idx.raw] for a, idx in
                                  zip(broadcasted_arrays, bidxes)]

            for aidx, abidx, skip_shape in zip(aidxes, a_broadcasted_idxs, skip_shapes):
                if skip_shape == broadcasted_skip_shape:
                    assert_equal(aidx, abidx)
                assert aidx.shape == skip_shape

            if skip_axes:
                # If there are skipped axes, recursively call iter_indices to
                # get each individual element of the resulting subarrays.
                for subidxes in iter_indices(*[x.shape for x in aidxes]):
                    items = [x[i.raw] for x, i in zip(aidxes, subidxes)]
                    # An empty array means the iteration would be skipped.
                    if any(a.size == 0 for a in items):
                        continue
                    vals.append(tuple(items))
            else:
                vals.append(aidxes)
    except ValueError as e:
        if "duplicate axes" in str(e):
            # There should be actual duplicate axes
            assert len({broadcasted_shape[i] for i in skip_axes}) < len(skip_axes)
            return
        raise # pragma: no cover

    assert len(set(vals)) == len(vals) == broadcasted_nitems

    # The indices should correspond to the values that would be matched up
    # if the arrays were broadcasted together.
    if not arrays:
        assert vals == [()]
    else:
        correct_vals = [tuple(i) for i in np.stack(broadcasted_arrays, axis=-1)
                        .reshape((broadcasted_nitems, len(arrays)))]
        # Also test that the indices are produced in a lexicographic order
        # (even though this isn't strictly guaranteed by the iter_indices
        # docstring) in the case when there are no skip axes. The order when
        # there are skip axes is more complicated because the skipped axes are
        # iterated together.
        if not skip_axes:
            assert vals == correct_vals
        else:
            assert set(vals) == set(correct_vals)

    assert n == nitems - 1

def test_iter_indices_errors():
    try:
        list(iter_indices((10,), skip_axes=(2,)))
    except AxisError as e:
        msg1 = str(e)
    else:
        raise RuntimeError("iter_indices did not raise AxisError") # pragma: no cover

    # Check that the message is the same one used by NumPy
    try:
        np.sum(np.arange(10), axis=2)
    except np.AxisError as e:
        msg2 = str(e)
    else:
        raise RuntimeError("np.sum() did not raise AxisError") # pragma: no cover

    assert msg1 == msg2

    try:
        list(iter_indices((2, 3), (3, 2)))
    except BroadcastError as e:
        msg1 = str(e)
    else:
        raise RuntimeError("iter_indices did not raise BroadcastError") # pragma: no cover

    # TODO: Check that the message is the same one used by NumPy
    # try:
    #     np.broadcast_shapes((2, 3), (3, 2))
    # except np.Error as e:
    #     msg2 = str(e)
    # else:
    #     raise RuntimeError("np.broadcast_shapes() did not raise AxisError") # pragma: no cover
    #
    # assert msg1 == msg2

@example(1, 1, 1)
@given(integers(0, 100), integers(0, 100), integers(0, 100))
def test_ncycles(i, n, m):
    N = ncycles(range(i), n)
    if n == 1:
        assert N == range(i)
    else:
        assert isinstance(N, ncycles)
        assert N.iterable == range(i)
        assert N.n == n
        assert f"range(0, {i})" in repr(N)
        assert str(n) in repr(N)

    L = list(N)
    assert len(L) == i*n
    for j in range(i*n):
        assert L[j] == j % i

    M = ncycles(N, m)
    if n*m == 1:
        assert M == range(i)
    else:
        assert isinstance(M, ncycles)
        assert M.iterable == range(i)
        assert M.n == n*m
