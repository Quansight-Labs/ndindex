import inspect

import numpy as np

from hypothesis import given, example, settings, assume
from hypothesis.strategies import one_of, tuples, none, integers

from pytest import raises, warns

from ..ndindex import ndindex, asshape, iter_indices
from ..booleanarray import BooleanArray
from ..integer import Integer
from ..ellipsis import ellipsis
from ..integerarray import IntegerArray
from ..tuple import Tuple
from .helpers import (ndindices, check_same, assert_equal, short_shapes, prod,
                      mutually_broadcastable_shapes)

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

@given(mutually_broadcastable_shapes,
       mutually_broadcastable_shapes.flatmap(
           lambda bs: one_of(none(), tuples(*(integers(-i, max(0, i-1)) for i in range(len(bs.result_shape)))))))
def test_iter_indices(broadcastable_shapes, skip_axes):
    shapes, result_shape = broadcastable_shapes

    if skip_axes is None:
        res = iter_indices(*shapes)
        skip_axes = ()
    else:
        res = iter_indices(*shapes, skip_axes=skip_axes)

    sizes = [prod(shape) for shape in shapes]
    ndim = len(result_shape)
    arrays = [np.arange(size).reshape(shape) for size, shape in zip(sizes, shapes)]

    normalized_skip_axes = sorted(ndindex(i).reduce(ndim).args[0] for i in skip_axes)
    # skip_shape = tuple(shape[i] for i in normalized_skip_axes)
    non_skip_shape = tuple(result_shape[i] for i in range(ndim) if i not in normalized_skip_axes)
    nitems = prod(non_skip_shape)

    vals = []
    n = -1
    try:
        for n, idxes in enumerate(res):
            assert len(idxes) == len(shapes)
            for idx, shape in zip(idxes, shapes):
                assert isinstance(idx, Tuple)
                assert len(idx.args) == len(shape)
                for i in range(len(idx.args)):
                    if i in normalized_skip_axes:
                        assert idx.args[i] == slice(None)
                    else:
                        assert isinstance(idx.args[i], Integer)


                # assert a[idx.raw].shape == skip_shape
                # assert set(a[idx.raw].flat).intersection(vals) == set()
                # vals.update(set(a[idx.raw].flat))
    except ValueError as e:
        # Handled in test_iter_indices_errors()
        if "duplicate axes" in str(e):
            assume(False)
        raise

    # assert vals == set(range(size))

    assert n == nitems - 1
