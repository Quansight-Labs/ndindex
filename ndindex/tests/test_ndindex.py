import inspect

from hypothesis import given, example

from pytest import raises

from ..ndindex import ndindex, isindex
from ..integer import Integer
from ..ellipsis import ellipsis
from .helpers import ndindices

from numpy import empty, ndarray

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

@given(ndindices())
def test_isindex(idx):
    assert isindex(idx, exclude=(type(idx),)) is False
    assert isindex(ndindex(idx)) is True

def test_isindex_ellipsis():
    assert isindex(Ellipsis) is True
    assert isindex(Ellipsis, exclude=(Ellipsis,)) is False
    raises(TypeError, lambda: isindex(ellipsis))

def test_isindex_ndarray():
    assert isindex(empty(1), exclude=(ndarray,)) is False
    raises(NotImplementedError, lambda: isindex(empty(1)))

def test_ndindex_ellipsis():
    raises(TypeError, lambda: ndindex(ellipsis))

def test_signature():
    sig = inspect.signature(Integer)
    assert sig.parameters.keys() == {'idx'}

@given(ndindices())
@example((1, ..., slice(1, 2)))
def test_str(idx):
    # The str form should be re-creatable
    index = ndindex(idx)
    d = {}
    exec("from ndindex import *", d)
    assert eval(str(index), d) == idx
