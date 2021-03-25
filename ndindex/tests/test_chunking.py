from itertools import zip_longest, tee

from numpy import arange, isin, sort, concatenate

from hypothesis import given, assume, example
from hypothesis.strategies import one_of

from pytest import raises

from ..chunking import ChunkSize
from ..tuple import Tuple
from ..ndindex import ndindex

from .helpers import assert_equal, chunk_sizes, chunk_shapes, prod, ints, slices, ndindices

def test_ChunkSize_constructor():
    raises(TypeError, lambda: ChunkSize(Tuple(1, 2, 3)))
    raises(TypeError, lambda: ChunkSize(1, 2, 3))
    raises(TypeError, lambda: ChunkSize(1))
    raises(ValueError, lambda: ChunkSize((-1, 2, 3)))
    raises(ValueError, lambda: ChunkSize((0, 2, 3)))
    raises(NotImplementedError, lambda: ChunkSize((None, 2, 3)))

@given(chunk_sizes())
def test_ChunkSize_eq(chunk_size_tuple):
    chunk_size = ChunkSize(chunk_size_tuple)
    new = type(chunk_size)(*chunk_size.args)

    assert chunk_size == chunk_size_tuple
    assert chunk_size_tuple == chunk_size
    assert new == chunk_size
    assert chunk_size == new
    assert new == chunk_size_tuple
    assert chunk_size_tuple == new

    assert hash(new) == hash(chunk_size)
    assert not (chunk_size == 'a')
    assert not ('a' == chunk_size)
    assert (chunk_size != 'a')
    assert ('a' != chunk_size)

    h = hash(chunk_size_tuple)
    assert hash(chunk_size) == h

@given(chunk_sizes(), one_of(ints(), slices()))
def test_ChunkSize_args(chunk_size_tuple, idx):
    chunk_size = ChunkSize(chunk_size_tuple)
    assert chunk_size.args == (chunk_size_tuple,)

    try:
        ndindex(idx)
    except ValueError:
        # Filter out invalid slices (TODO: do this in the strategy)
        assume(False)

    # Should index the same way
    # TODO: Refactor check_same() so we can use that
    try:
        chunk_size_idx = chunk_size[idx]
    except IndexError:
        try:
            tuple_idx = chunk_size_tuple[idx]
        except IndexError:
            pass
        else:
            raise AssertionError("ChunkSize raised but tuple did not")
    else:
        tuple_idx = chunk_size_tuple[idx]
        assert chunk_size_idx == tuple_idx

@given(chunk_sizes())
def test_ChunkSize_tuple(chunk_size_tuple):
    # Test that ChunkSize behaves like a tuple
    chunk_size = ChunkSize(chunk_size_tuple)
    assert tuple(chunk_size) == chunk_size_tuple

def test_indices_error():
    raises(ValueError, lambda: next(ChunkSize((1, 2)).indices((1, 2, 3))))

@given(chunk_sizes(), chunk_shapes)
def test_num_chunks(chunk_size, shape):
    chunk_size = ChunkSize(chunk_size)
    assert chunk_size.num_chunks(shape) == len(list(chunk_size.indices(shape)))

@given(chunk_sizes(), chunk_shapes)
def test_indices(chunk_size, shape):
    chunk_size = ChunkSize(chunk_size)
    indices = chunk_size.indices(shape)
    size = prod(shape)
    a = arange(size).reshape(shape)

    subarrays = []
    for idx in indices:
        # The indices should be fully expanded
        assert idx.expand(shape) == idx
        # Except for indices at the edges, they should index a full chunk
        if not any(s.stop == i for s, i in zip(idx.args, shape)):
            assert idx.newshape(shape) == chunk_size
        # Make sure they can be indexed
        subarrays.append(a[idx.raw])
    # Check that indices together index every element of the array exactly
    # once.
    elements = [i for x in subarrays for i in x.flatten()]
    assert sorted(elements) == list(range(size))

def test_as_subchunks_error():
    raises(ValueError, lambda: next(ChunkSize((1, 2)).as_subchunks(..., (1, 2, 3))))

@example((1,), True, (1,))
@example(chunk_size=(1, 1), idx=slice(1, None, 2), shape=(4, 1))
@example((1,), ..., (0,))
@example((2, 2), (0, 3), (5, 5))
@example((2, 2), (slice(0, 5, 2), slice(0, 5, 3)), (5, 5))
@example((2, 2), ([0, 0],), (5, 5))
@given(chunk_sizes(), ndindices, chunk_shapes)
def test_as_subchunks(chunk_size, idx, shape):
    chunk_size = ChunkSize(chunk_size)
    size = prod(shape)
    a = arange(size).reshape(shape)
    idx = ndindex(idx)

    try:
        idx.reduce(shape)
    except IndexError:
        assume(False)

    full_idx = a[idx.raw]

    subarrays = []
    fast = chunk_size.as_subchunks(idx, shape)
    slow = chunk_size.as_subchunks(idx, shape, _force_slow=True)
    slow2 = chunk_size.as_subchunks(idx, shape, _force_slow=True)
    no_fallback = chunk_size.as_subchunks(idx, shape, _force_slow=False)
    slow_raised_notimplementederror = False
    try:
        next(slow2)
    except StopIteration:
        pass
    except NotImplementedError:
        # The fallback isn't implemented, but the fast case may still be.
        slow, fast = tee(fast, 2)
        slow_raised_notimplementederror = True
    if not slow_raised_notimplementederror:
        # If it works (no NotImplementedError), it shouldn't use the fallback.
        try:
            next(no_fallback)
        except StopIteration:
            pass
    try:
        for c, cslow in zip_longest(fast, slow):
            assert c == cslow
            index = idx.expand(shape).as_subindex(c)
            chunk = a[c.raw]
            subchunk = chunk[index.raw]
            # Not empty
            assert subchunk.size > 0
            # Indexes the right elements (c.f. test_as_subindex)
            assert_equal(subchunk.flatten(), full_idx[isin(full_idx, chunk)])
            subarrays.append(subchunk)
    except NotImplementedError:
        # NotImplementedError should only be allowed from the fallback algorithm
        if not slow_raised_notimplementederror:
            raise
        return

    # Picks all elements
    if subarrays:
        elements = concatenate([x.flatten() for x in subarrays])
    else:
        elements = arange(0)
    assert_equal(sort(elements), sort(full_idx.flatten()))

def test_num_subchunks_error():
    raises(ValueError, lambda: next(ChunkSize((1, 2)).num_subchunks(..., (1, 2, 3))))

@example((1,), True, (1,))
@example(chunk_size=(1, 1), idx=slice(1, None, 2), shape=(4, 1))
@example((1,), ..., (0,))
@example((2, 2), (0, 3), (5, 5))
@example((2, 2), (slice(0, 5, 2), slice(0, 5, 3)), (5, 5))
@example((2, 2), ([0, 0],), (5, 5))
@given(chunk_sizes(), ndindices, chunk_shapes)
def test_num_subchunks(chunk_size, idx, shape):
    chunk_size = ChunkSize(chunk_size)
    idx = ndindex(idx)

    try:
        idx.reduce(shape)
    except IndexError:
        assume(False)

    subchunks = chunk_size.as_subchunks(idx, shape)
    try:
        actual_num_subchunks = len(list(subchunks))
        computed_num_subchunks = chunk_size.num_subchunks(idx, shape)
    except NotImplementedError:
        return
    assert computed_num_subchunks == actual_num_subchunks
