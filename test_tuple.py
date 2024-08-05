from itertools import product

from simple_tuple import SimpleTuple as Tuple

from pytest import raises

from hypothesis import given

from numpy import arange

from ndindex.tests.helpers import iterslice, check_same, Tuples, short_shapes, prod

def test_tuple_constructor():
    # Test things in the Tuple constructor that are not tested by the other
    # tests below.
    raises(ValueError, lambda: Tuple((1, 2, 3)))
    raises(ValueError, lambda: Tuple(0, (1, 2, 3)))

    # Test NotImplementedError behavior for Tuples with arrays split up by
    # slices, ellipses, and newaxes.
    raises(NotImplementedError, lambda: Tuple(0, slice(None), [0]))
    raises(NotImplementedError, lambda: Tuple([0], slice(None), [0]))
    raises(NotImplementedError, lambda: Tuple([0], slice(None), [0]))
    raises(NotImplementedError, lambda: Tuple(0, ..., [0]))
    raises(NotImplementedError, lambda: Tuple([0], ..., [0]))
    raises(NotImplementedError, lambda: Tuple([0], ..., [0]))
    raises(NotImplementedError, lambda: Tuple(0, None, [0]))
    raises(NotImplementedError, lambda: Tuple([0], None, [0]))
    raises(NotImplementedError, lambda: Tuple([0], None, [0]))
    # Make sure this doesn't raise
    Tuple(0, slice(None), 0)
    Tuple(0, ..., 0)
    Tuple(0, None, 0)

def test_tuple_exhaustive():
    # Exhaustive tests here have to be very limited because of combinatorial
    # explosion.
    a = arange(2*2*2).reshape((2, 2, 2))
    types = {
        slice: lambda: iterslice((-1, 1), (-1, 1), (-1, 1), one_two_args=False),
        # slice: _iterslice,
        int: lambda: ((i,) for i in range(-3, 3)),
        type(...): lambda: ()
    }

    for t1, t2, t3 in product(types, repeat=3):
        for t1_args in types[t1]():
            for t2_args in types[t2]():
                for t3_args in types[t3]():
                    idx1 = t1(*t1_args)
                    idx2 = t2(*t2_args)
                    idx3 = t3(*t3_args)

                    idx = (idx1, idx2, idx3)
                    # Disable the same exception check because there could be
                    # multiple invalid indices in the tuple, and for instance
                    # numpy may give an IndexError but we would give a
                    # TypeError because we check the type first.
                    check_same(a, idx, same_exception=False)

@given(Tuples, short_shapes)
def test_tuples_hypothesis(t, shape):
    a = arange(prod(shape)).reshape(shape)
    check_same(a, t, same_exception=False)
