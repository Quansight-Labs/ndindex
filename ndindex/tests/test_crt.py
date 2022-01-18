from .._crt import crt

from hypothesis import given
from hypothesis.strategies import integers, lists, shared

size = shared(integers(min_value=1, max_value=10))
@given(
    size.flatmap(lambda s: lists(integers(min_value=1), min_size=s, max_size=s)),
    size.flatmap(lambda s: lists(integers(), min_size=s, max_size=s)),
)
def test_crt(m, v):
    res = crt(m, v)

    if res is None:
        # The algorithm gives no solution. Not sure how to test this.
        return

    for m_i, v_i in zip(m, v):
        assert v_i % m_i == res % m_i
