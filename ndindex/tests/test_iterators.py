import numpy as np

from hypothesis import assume, given, example
from hypothesis.strategies import (one_of, integers, tuples as
                                   hypothesis_tuples, just, lists, shared)

from pytest import raises

from ..ndindex import ndindex
from ..iterators import (iter_indices, ncycles, BroadcastError,
                            AxisError, broadcast_shapes, remove_indices,
                            unremove_indices, associated_axis)
from ..integer import Integer
from ..tuple import Tuple
from .helpers import (prod, mutually_broadcastable_shapes_with_skipped_axes,
                      skip_axes_st, mutually_broadcastable_shapes, tuples,
                      shapes)

@example([((1, 1), (1, 1)), (None, 1)], (0,))
@example([((0,), (0,)), (None,)], (0,))
@example([((1, 2), (2, 1)), (2, None)], 1)
@given(mutually_broadcastable_shapes_with_skipped_axes(), skip_axes_st)
def test_iter_indices(broadcastable_shapes, skip_axes):
    # broadcasted_shape will contain None on the skip_axes, as those axes
    # might not be broadcast compatible
    shapes, broadcasted_shape = broadcastable_shapes
    # We need no more than 31 dimensions so that the np.stack call below
    # doesn't fail.
    assume(len(broadcasted_shape) < 32)

    # 1. Normalize inputs
    _skip_axes = (skip_axes,) if isinstance(skip_axes, int) else skip_axes

    # Double check the mutually_broadcastable_shapes_with_skipped_axes
    # strategy
    for i in _skip_axes:
        assert broadcasted_shape[i] is None

    # Skipped axes may not be broadcast compatible. Since the index for a
    # skipped axis should always be a slice(None), the result should be the
    # same if the skipped axes are all moved to the end of the shape.
    canonical_shapes = []
    for s in shapes:
        c = remove_indices(s, _skip_axes)
        c = c + tuple(1 for i in _skip_axes)
        canonical_shapes.append(c)
    canonical_skip_axes = list(range(-1, -len(_skip_axes) - 1, -1))
    broadcasted_canonical_shape = list(broadcast_shapes(*canonical_shapes,
                                                        skip_axes=canonical_skip_axes))
    for i in range(len(broadcasted_canonical_shape)):
        if broadcasted_canonical_shape[i] is None:
            broadcasted_canonical_shape[i] = 1

    skip_shapes = [tuple(1 for i in _skip_axes) for shape in shapes]
    non_skip_shapes = [remove_indices(shape, skip_axes) for shape in shapes]
    broadcasted_non_skip_shape = remove_indices(broadcasted_shape, skip_axes)
    assert None not in broadcasted_non_skip_shape
    assert broadcast_shapes(*non_skip_shapes) == broadcasted_non_skip_shape

    nitems = prod(broadcasted_non_skip_shape)

    if _skip_axes == ():
        res = iter_indices(*shapes)
        broadcasted_res = iter_indices(broadcast_shapes(*shapes))
    else:
        res = iter_indices(*shapes, skip_axes=skip_axes)
        broadcasted_res = iter_indices(broadcasted_canonical_shape,
                                       skip_axes=canonical_skip_axes)

    sizes = [prod(shape) for shape in shapes]
    arrays = [np.arange(size).reshape(shape) for size, shape in zip(sizes, shapes)]
    canonical_sizes = [prod(shape) for shape in canonical_shapes]
    canonical_arrays = [np.arange(size).reshape(shape) for size, shape in zip(canonical_sizes, canonical_shapes)]

    # 2. Check that iter_indices is the same whether or not the shapes are
    # broadcasted together first. Also check that every iterated index is the
    # expected type and there are as many as expected.
    vals = []
    n = -1

    def _move_slices_to_end(idx):
        assert isinstance(idx, Tuple)
        idx2 = list(idx.args)
        slices = [i for i in range(len(idx2)) if idx2[i] == slice(None)]
        idx2 = remove_indices(idx2, slices)
        idx2 = idx2 + (slice(None),)*len(slices)
        return Tuple(*idx2)

    for n, (idxes, bidxes) in enumerate(zip(res, broadcasted_res)):
        assert len(idxes) == len(shapes)
        assert len(bidxes) == 1
        for idx, shape in zip(idxes, shapes):
            assert isinstance(idx, Tuple)
            assert len(idx.args) == len(shape)

            normalized_skip_axes = sorted(ndindex(i).reduce(len(shape)).raw for i in _skip_axes)
            for i in range(len(idx.args)):
                if i in normalized_skip_axes:
                    assert idx.args[i] == slice(None)
                else:
                    assert isinstance(idx.args[i], Integer)

        canonical_idxes = [_move_slices_to_end(idx) for idx in idxes]
        a_indexed = tuple([a[idx.raw] for a, idx in zip(arrays, idxes)])
        canonical_a_indexed = tuple([a[idx.raw] for a, idx in
                                  zip(canonical_arrays, canonical_idxes)])

        for c_indexed, skip_shape in zip(canonical_a_indexed, skip_shapes):
            assert c_indexed.shape == skip_shape

        if _skip_axes:
            # If there are skipped axes, recursively call iter_indices to
            # get each individual element of the resulting subarrays.
            for subidxes in iter_indices(*[x.shape for x in canonical_a_indexed]):
                items = [x[i.raw] for x, i in zip(canonical_a_indexed, subidxes)]
                vals.append(tuple(items))
        else:
            vals.append(a_indexed)

    # assert both iterators have the same length
    raises(StopIteration, lambda: next(res))
    raises(StopIteration, lambda: next(broadcasted_res))

    # Check that the correct number of items are iterated
    assert n == nitems - 1
    assert len(set(vals)) == len(vals) == nitems

    # 3. Check that every element of the (broadcasted) arrays is represented
    # by an iterated index.

    # The indices should correspond to the values that would be matched up
    # if the arrays were broadcasted together.
    if not arrays:
        assert vals == [()]
    else:
        correct_vals = [tuple(i) for i in np.stack(np.broadcast_arrays(*canonical_arrays), axis=-1).reshape((nitems, len(arrays)))]
        # Also test that the indices are produced in a lexicographic order
        # (even though this isn't strictly guaranteed by the iter_indices
        # docstring) in the case when there are no skip axes. The order when
        # there are skip axes is more complicated because the skipped axes are
        # iterated together.
        if not _skip_axes:
            assert vals == correct_vals
        else:
            assert set(vals) == set(correct_vals)

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

    try:
        np.broadcast_shapes((2, 3), (3, 2))
    except ValueError as e:
        msg2 = str(e)
    else:
        raise RuntimeError("np.broadcast_shapes() did not raise ValueError") # pragma: no cover

    assert msg1 == msg2

    raises(NotImplementedError, lambda: list(iter_indices((1, 2), skip_axes=(0, -1))))

    with raises(ValueError, match=r"duplicate axes"):
        list(iter_indices((1, 2), skip_axes=(0, 1, 0)))

    raises(TypeError, lambda: list(iter_indices(1, 2)))
    raises(TypeError, lambda: list(iter_indices(1, 2, (2, 2))))
    raises(TypeError, lambda: list(iter_indices([(1, 2), (2, 2)])))

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


@given(lists(shapes, max_size=32))
def test_broadcast_shapes_errors(shapes):
    error = True
    try:
        broadcast_shapes(*shapes)
    except BroadcastError as exc:
        e = exc
    else:
        error = False

    # The ndindex and numpy errors won't match in general, because
    # ndindex.broadcast_shapes gives an error with the first two shapes that
    # aren't broadcast compatible, but numpy doesn't always, due to different
    # implementation algorithms (e.g., the message from
    # np.broadcast_shapes((0,), (0, 2), (2, 0)) mentions the last two shapes
    # whereas ndindex.broadcast_shapes mentions the first two).

    # Instead, just confirm that the error message is correct as stated, and
    # check against the numpy error message when just broadcasting the two
    # reportedly bad shapes.

    if not error:
        try:
            np.broadcast_shapes(*shapes)
        except:
            raise RuntimeError("ndindex.broadcast_shapes raised but np.broadcast_shapes did not")
        return

    assert shapes[e.arg1] == e.shape1
    assert shapes[e.arg2] == e.shape2

    try:
        np.broadcast_shapes(e.shape1, e.shape2)
    except ValueError as np_exc:
        # Check that they do in fact not broadcast, and the error messages are
        # the same modulo the different arg positions.
        assert str(BroadcastError(0, e.shape1, 1, e.shape2)) == str(np_exc)
    else:
        raise RuntimeError("ndindex.broadcast_shapes raised but np.broadcast_shapes did not")

    raises(TypeError, lambda: broadcast_shapes(1, 2))
    raises(TypeError, lambda: broadcast_shapes(1, 2, (2, 2)))
    raises(TypeError, lambda: broadcast_shapes([(1, 2), (2, 2)]))

@given(mutually_broadcastable_shapes_with_skipped_axes(), skip_axes_st)
def test_broadcast_shapes_skip_axes(broadcastable_shapes, skip_axes):
    shapes, broadcasted_shape = broadcastable_shapes
    assert broadcast_shapes(*shapes, skip_axes=skip_axes) == broadcasted_shape

@given(mutually_broadcastable_shapes, lists(integers(-20, 20), max_size=20))
def test_broadcast_shapes_skip_axes_errors(broadcastable_shapes, skip_axes):
    shapes, broadcasted_shape = broadcastable_shapes
    if any(i < 0 for i in skip_axes) and any(i >= 0 for i in skip_axes):
        raises(NotImplementedError, lambda: broadcast_shapes(*shapes, skip_axes=skip_axes))
        return

    # Test that broadcast_shapes raises IndexError when skip_axes are out of
    # bounds
    try:
        if not shapes and skip_axes:
            raise IndexError
        for shape in shapes:
            for i in skip_axes:
                shape[i]
    except IndexError:
        indexerror = True
    else:
        indexerror = False

    broadcasterror = False
    try:
        broadcast_shapes(*shapes, skip_axes=skip_axes)
    except IndexError:
        if not indexerror:
            raise RuntimeError("broadcast_shapes raised IndexError but should not have")
    except BroadcastError:
        broadcasterror = True
    else:
        if indexerror:
            raise RuntimeError("broadcast_shapes did not raise IndexError but should have")

    if not indexerror:
        broadcastable = [remove_indices(shape, skip_axes) for shape in shapes]
        try:
            np.broadcast_shapes(*broadcastable)
        except ValueError:
            if not broadcasterror:
                raise RuntimeError("broadcast_shapes did not raise BroadcastError but should have")
        else:
            if broadcasterror:
                raise RuntimeError("broadcast_shapes raised BroadcastError but should not have")

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
    _skip_axes = (skip_axes,) if isinstance(skip_axes, int) else skip_axes

    for shape in shapes:
        assert None not in shape
    for i in _skip_axes:
        assert broadcasted_shape[i] is None

    _shapes = [remove_indices(shape, skip_axes) for shape in shapes]
    _broadcasted_shape = remove_indices(broadcasted_shape, skip_axes)

    assert None not in _broadcasted_shape
    assert broadcast_shapes(*_shapes) == _broadcasted_shape

@example([[(0, 10, 2, 3, 10, 4), (1, 10, 1, 0, 10, 2, 3, 4)],
          (1, None, 1, 0, None, 2, 3, 4)], (1, 4))
@example([[(2, 0, 3, 4)], (2, None, 3, 4)], (1,))
@example([[(0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)], (0, None, None, 0, 0, 0)], (1, 2))
@given(mutually_broadcastable_shapes_with_skipped_axes(), skip_axes_st)
def test_associated_axis(broadcastable_shapes, skip_axes):
    _skip_axes = (skip_axes,) if isinstance(skip_axes, int) else skip_axes

    shapes, broadcasted_shape = broadcastable_shapes
    ndim = len(broadcasted_shape)

    normalized_skip_axes = [ndindex(i).reduce(ndim) for i in _skip_axes]

    for shape in shapes:
        n = len(shape)
        for i in range(-len(shape), 0):
            val = shape[i]

            idx = associated_axis(shape, broadcasted_shape, i, _skip_axes)
            bval = broadcasted_shape[idx]
            if bval is None:
                if _skip_axes[0] >= 0:
                    assert ndindex(i).reduce(n) == ndindex(idx).reduce(ndim) in normalized_skip_axes
                else:
                    assert ndindex(i).reduce(n).raw - n == \
                        ndindex(idx).reduce(ndim).raw - ndim in _skip_axes
            else:
                assert val == 1 or bval == val
