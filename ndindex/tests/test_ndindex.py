import inspect

from hypothesis import given

from ..ndindex import ndindex
from ..integer import Integer
from .helpers import ndindices

@given(ndindices())
def test_eq(idx):
    new = type(idx)(*idx.args)
    assert (new == idx) is True
    assert (new.raw == idx.raw) is True
    assert hash(new) == hash(idx)
    assert (idx == idx.raw) is True
    assert (idx.raw == idx) is True
    assert (idx == 'a') is False
    assert ('a' == idx) is False
    assert (idx != 'a') is True
    assert ('a' != idx) is True

@given(ndindices())
def test_ndindex(idx):
    assert ndindex(idx) == idx
    assert ndindex(idx).raw == idx
    ix = ndindex(idx)
    assert ndindex(ix.raw) == ix

def test_signature():
    sig = inspect.signature(Integer)
    assert sig.parameters.keys() == {'idx'}
