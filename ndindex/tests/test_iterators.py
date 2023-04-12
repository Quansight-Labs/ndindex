import numpy as np

from hypothesis import assume, given, example
from hypothesis.strategies import (one_of, integers, tuples as
                                   hypothesis_tuples, just, lists, shared)

from pytest import raises

from ..ndindex import ndindex
from ..iterators import (iter_indices, ncycles, BroadcastError,
                            AxisError, broadcast_shapes, remove_indices,
                            unremove_indices)
from ..integer import Integer
from ..tuple import Tuple
from .helpers import (assert_equal, prod,
                      mutually_broadcastable_shapes_with_skipped_axes,
                      skip_axes_st, mutually_broadcastable_shapes, tuples,
                      shapes)

@example([((1, 1), (1, 1)), (None, 1)], (0, 0))
@example([((), (0,)), (None,)], (0,))
@example([((1, 2), (2, 1)), (2, None)], 1)
@given(mutually_broadcastable_shapes_with_skipped_axes(), skip_axes_st)
def test_iter_indices(broadcastable_shapes, _skip_axes):
    # broadcasted_shape will contain None on the skip_axes, as those axes
    # might not be broadcast compatible
    shapes, broadcasted_shape = broadcastable_shapes

    # 1. Normalize inputs
    skip_axes = (_skip_axes,) if isinstance(_skip_axes, int) else () if _skip_axes is None else _skip_axes
    ndim = len(broadcasted_shape)

    # Double check the mutually_broadcastable_shapes_with_skipped_axes
    # strategy
    for i in skip_axes:
        assert broadcasted_shape[i] is None

    # Use negative indices to index the skip axes since only shapes that have
    # the skip axis will include a slice.
    normalized_skip_axes = sorted(ndindex(i).reduce(ndim).args[0] - ndim for i in skip_axes)
    canonical_shapes = [list(s) for s in shapes]
    for i in normalized_skip_axes:
        for s in canonical_shapes:
            if ndindex(i).isvalid(len(s)):
                s[i] = 1
    skip_shapes = [tuple(shape[i] for i in normalized_skip_axes if ndindex(i).isvalid(len(shape))) for shape in canonical_shapes]
    broadcasted_skip_shape = tuple(broadcasted_shape[i] for i in normalized_skip_axes)

    broadcasted_non_skip_shape = tuple(broadcasted_shape[i] for i in range(-1, -ndim-1, -1) if i not in normalized_skip_axes)
    nitems = prod(broadcasted_non_skip_shape)

    if _skip_axes is None:
        res = iter_indices(*shapes)
        broadcasted_res = iter_indices(np.broadcast_shapes(*shapes))
    else:
        # Skipped axes may not be broadcast compatible. Since the index for a
        # skipped axis should always be a slice(None), the result should be
        # the same if the skipped axes are all replaced with 1.
        res = iter_indices(*shapes, skip_axes=_skip_axes)
        broadcasted_res = iter_indices(np.broadcast_shapes(*canonical_shapes),
                                       skip_axes=_skip_axes)

    sizes = [prod(shape) for shape in shapes]
    arrays = [np.arange(size).reshape(shape) for size, shape in zip(sizes, shapes)]
    canonical_sizes = [prod(shape) for shape in canonical_shapes]
    canonical_arrays = [np.arange(size).reshape(shape) for size, shape in zip(canonical_sizes, canonical_shapes)]
    broadcasted_arrays = np.broadcast_arrays(*canonical_arrays)

    # 2. Check that iter_indices is the same whether or not the shapes are
    # broadcasted together first. Also check that every iterated index is the
    # expected type and there are as many as expected.
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
            canonical_aidxes = tuple([a[idx.raw] for a, idx in zip(canonical_arrays, idxes)])
            a_broadcasted_idxs = [a[idx.raw] for a, idx in
                                  zip(broadcasted_arrays, bidxes)]

            for aidx, abidx, skip_shape in zip(canonical_aidxes, a_broadcasted_idxs, skip_shapes):
                if skip_shape == broadcasted_skip_shape:
                    assert_equal(aidx, abidx)
                assert aidx.shape == skip_shape

            if skip_axes:
                # If there are skipped axes, recursively call iter_indices to
                # get each individual element of the resulting subarrays.
                for subidxes in iter_indices(*[x.shape for x in canonical_aidxes]):
                    items = [x[i.raw] for x, i in zip(canonical_aidxes, subidxes)]
                    vals.append(tuple(items))
            else:
                vals.append(aidxes)
    except ValueError as e:
        if "duplicate axes" in str(e):
            # There should be actual duplicate axes
            assert len({broadcasted_shape[i] for i in skip_axes}) < len(skip_axes)
            return
        raise # pragma: no cover

    assert len(set(vals)) == len(vals) == nitems

    # 3. Check that every element of the (broadcasted) arrays is represented
    # by an iterated index.

    # The indices should correspond to the values that would be matched up
    # if the arrays were broadcasted together.
    if not arrays:
        assert vals == [()]
    else:
        correct_vals = [tuple(i) for i in np.stack(broadcasted_arrays, axis=-1).reshape((nitems, len(arrays)))]
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

    raises(NotImplementedError, lambda: list(iter_indices((1, 2), skip_axes=(0, -1))))
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

@given(one_of(mutually_broadcastable_shapes,
              hypothesis_tuples(tuples(shapes), just(None))))
def test_broadcast_shapes(broadcastable_shapes):
    shapes, broadcasted_shape = broadcastable_shapes
    if broadcasted_shape is not None:
        assert broadcast_shapes(*shapes) == broadcasted_shape

    arrays = [np.empty(shape) for shape in shapes]
    broadcastable = True
    try:
        broadcasted_shape = np.broadcast(*arrays).shape
    except ValueError:
        broadcastable = False

    if broadcastable:
        assert broadcast_shapes(*shapes) == broadcasted_shape
    else:
        raises(BroadcastError, lambda: broadcast_shapes(*shapes))

remove_indices_n = shared(integers(0, 100))

@given(remove_indices_n,
       remove_indices_n.flatmap(lambda n: lists(integers(-n, n), unique=True)))
def test_remove_indices(n, idxes):
    if idxes:
        assume(max(idxes) < n)
        assume(min(idxes) >= -n)
    a = tuple(range(n))
    b = remove_indices(a, idxes)

    A = list(a)
    for i in idxes:
        A[i] = None

    assert set(A) - set(b) == ({None} if idxes else set())
    assert set(b) - set(A) == set()

    # Check the order is correct
    j = 0
    for i in range(n):
        val = A[i]
        if val == None:
            assert val not in b
        else:
            assert b[j] == val
            j += 1

    # Test that unremove_indices is the inverse
    if all(i >= 0 for i in idxes) or all(i < 0 for i in idxes):
        assert unremove_indices(b, idxes) == tuple(A)
    else:
        raises(NotImplementedError, lambda: unremove_indices(b, idxes))

# Meta-test for the hypothesis strategy
@given(mutually_broadcastable_shapes_with_skipped_axes(), skip_axes_st)
def test_mutually_broadcastable_shapes_with_skipped_axes(broadcastable_shapes,
                                                         skip_axes):
    shapes, broadcasted_shape = broadcastable_shapes
    _skip_axes = (skip_axes,) if isinstance(skip_axes, int) else ()

    for shape in shapes:
        assert None not in shape
    for i in _skip_axes:
        assert broadcasted_shape[i] is None

    _shapes = [remove_indices(shape, skip_axes) for shape in shapes]
    _broadcasted_shape = remove_indices(broadcasted_shape, skip_axes)

    assert None not in _broadcasted_shape
    assert broadcast_shapes(*_shapes) == _broadcasted_shape
