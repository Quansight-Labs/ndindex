from hypothesis import given

from ..chunking import ChunkSize

from .helpers import chunk_sizes, shapes

@given(chunk_sizes, shapes)
def test_num_chunks(chunk_size, shape):
    chunk_size = ChunkSize(chunk_size)
    assert chunk_size.num_chunks(shape) == len(chunk_size.indices(shape))
