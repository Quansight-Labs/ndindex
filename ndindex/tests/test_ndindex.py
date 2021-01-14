import inspect

import numpy as np

from hypothesis import given, example, settings

from pytest import raises, warns

from ..ndindex import ndindex, asshape
from ..booleanarray import BooleanArray
from ..integer import Integer
from ..ellipsis import ellipsis
from ..integerarray import IntegerArray
from ..tuple import Tuple
from .helpers import ndindices, check_same, assert_equal

@given(ndindices)
def test_eq(idx):
    index = ndindex(idx)
    new = type(index)(*index.args)

    if isinstance(new.raw, np.ndarray):
        # trying to get a single value out of comparing two arrays requires all sorts of special handling, just let numpy do it
        assert np.array_equal(new.raw, index.raw)
    else:
        assert new.raw == index.raw

    assert (new == index)
    assert (new.raw == index)
    assert (new == index.raw)
    assert (index == new)
    assert (index.raw == new)
    assert (index == new.raw)

    assert (index.raw == index)
    assert hash(new) == hash(index)
    assert (index == index.raw)
    assert not (index == 'a')
    assert not ('a' == index)
    assert (index != 'a')
    assert ('a' != index)

    try:
        h = hash(idx)
    except TypeError:
        pass
    else:
        assert hash(index) == h

    try:
        h = hash(index.raw)
    except TypeError:
        pass
    else:
        assert hash(index) == h

def test_eq_array_raises():
    index = ndindex([1, 2, 3])
    with raises(TypeError):
        np.equal(index.raw, index)
    with raises(TypeError):
        np.array_equal(index.raw, index)

def test_eq_explicit():
    assert Integer(0) != False
    assert Integer(1) != True
    assert Integer(0) != IntegerArray(0)
    assert IntegerArray([0, 1]) != [False, True]
    assert IntegerArray([0, 1]) == [0, 1]
    assert BooleanArray([False, True]) != [0, 1]
    assert BooleanArray([False, True]) == [False, True]

@example((np.array([1, 2]), 0))
@example([1, 2, 3])
@given(ndindices)
def test_ndindex(idx):
    index = ndindex(idx)
    assert index == idx
    def test_raw_eq(idx, index):
        if isinstance(idx, np.ndarray):
            assert_equal(index.raw, idx)
        elif isinstance(idx, list):
            assert index.dtype in [np.intp, np.bool_]
            assert_equal(index.raw, np.asarray(idx, dtype=index.dtype))
        elif isinstance(idx, tuple):
            assert type(index.raw) == type(idx)
            assert len(index.raw) == len(idx)
            assert index.args == index.raw
            for i, j in zip(idx, index.args):
                test_raw_eq(i, j)
        else:
            assert index.raw == idx
    test_raw_eq(idx, index)
    assert ndindex(index.raw) == index

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


@example(([0, 1],))
@example((IntegerArray([], (0, 1)),))
@example(IntegerArray([], (0, 1)))
@example((1, ..., slice(1, 2)))
# eval can sometimes be slower than the default deadline of 200ms for large
# array indices
@settings(deadline=None)
@given(ndindices)
def test_repr_str(idx):
    # The repr form should be re-creatable
    index = ndindex(idx)
    d = {}
    exec("from ndindex import *", d)
    assert eval(repr(index), d) == idx

    # Str may not be re-creatable. Just test that it doesn't give an exception.
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
    raises(TypeError, lambda: asshape((True,)))
