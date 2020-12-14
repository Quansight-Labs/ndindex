from numpy import arange

from hypothesis import given

from ..chunking import ChunkSize

from .helpers import chunk_sizes, chunk_shapes, prod

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

@given(chunk_sizes(), chunk_shapes)
def test_num_chunks(chunk_size, shape):
    chunk_size = ChunkSize(chunk_size)
    assert chunk_size.num_chunks(shape) == len(list(chunk_size.indices(shape)))
