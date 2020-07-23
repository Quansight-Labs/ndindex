import inspect

import numpy as np

from hypothesis import given, example, settings

from pytest import raises, warns

from ..ndindex import ndindex, asshape
from ..integer import Integer
from ..ellipsis import ellipsis
from ..integerarray import IntegerArray
from ..tuple import Tuple
from .helpers import ndindices, check_same, assert_equal

@given(ndindices())
def test_eq(idx):
    index = ndindex(idx)
    new = type(index)(*index.args)
    assert (new == index) is True
    try:
        if isinstance(new.raw, np.ndarray):
            raise ValueError
        assert (new.raw == index.raw) is True
        assert (index.raw == index) is True
    except ValueError:
        np.testing.assert_equal(new.raw, index.raw)
        # Sadly, there is now way to bypass array.__eq__ from producing an
        # array.
    assert hash(new) == hash(index)
    assert (index == index.raw) is True
    assert (index == 'a') is False
    assert ('a' == index) is False
    assert (index != 'a') is True
    assert ('a' != index) is True

@given(ndindices())
def test_ndindex(idx):
    index = ndindex(idx)
    assert index == idx
    if isinstance(idx, np.ndarray):
        assert_equal(index.raw, idx)
    elif isinstance(idx, list):
        assert_equal(index.raw, np.asarray(idx, dtype=np.intp))
    else:
        assert index.raw == idx
    assert ndindex(index.raw) == index

def test_ndindex_not_implemented():
    a = np.arange(10)
    for idx in [np.array([True, False]*5), True, False, None]:
        raises(NotImplementedError, lambda: ndindex(idx))
        # Make sure the index really is valid
        a[idx]

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

@example(IntegerArray([], (0, 1)))
@example((1, ..., slice(1, 2)))
# eval can sometimes be slower than the default deadline of 200ms for large
# array indices
@settings(deadline=None)
@given(ndindices())
def test_repr(idx):
    # The repr form should be re-creatable
    index = ndindex(idx)
    d = {}
    exec("from ndindex import *", d)
    assert eval(repr(index), d) == idx

@given(ndindices())
def test_str(idx):
    # Str may not be re-creatable. Just test that it doesn't give an exception.
    index = ndindex(idx)
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
