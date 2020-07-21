import inspect

import numpy as np

from hypothesis import given, example

from pytest import raises

from ..ndindex import ndindex, asshape
from ..integer import Integer
from ..ellipsis import ellipsis
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

def test_ndindex_not_implemented():
    a = np.arange(10)
    for idx in [[], [1, 2], np.array([1, 2]), np.array([True, False]*5), True,
                False, None]:
        raises(NotImplementedError, lambda: ndindex(idx))
        # Make sure the index really is valid
        a[idx]

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
