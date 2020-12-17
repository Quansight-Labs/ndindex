import sys
from itertools import chain
from functools import reduce
from operator import mul

from numpy import intp, bool_, array
import numpy.testing

from pytest import fail

from hypothesis.strategies import (integers, none, one_of, lists, just,
                                   builds, shared, composite)
from hypothesis.extra.numpy import arrays

from ..ndindex import ndindex

# Hypothesis strategies for generating indices. Note that some of these
# strategies are nominally already defined in hypothesis, but we redefine them
# here because the hypothesis definitions are too restrictive. For example,
# hypothesis's slices strategy does not generate slices with negative indices.
# Similarly, hypothesis.extra.numpy.basic_indices only generates tuples.

# np.prod has overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)

nonnegative_ints = integers(0, 10)
negative_ints = integers(-10, -1)
ints = lambda: one_of(nonnegative_ints, negative_ints)

def slices(start=one_of(none(), ints()), stop=one_of(none(), ints()),
           step=one_of(none(), ints())):
    return builds(slice, start, stop, step)

ellipses = lambda: just(...)
newaxes = lambda: just(None)

# hypotheses.strategies.tuples only generates tuples of a fixed size
def tuples(elements, *, min_size=0, max_size=None, unique_by=None, unique=False):
    return lists(elements, min_size=min_size, max_size=max_size,
                 unique_by=unique_by, unique=unique).map(tuple)

shapes = tuples(integers(0, 10)).filter(
             # numpy gives errors with empty arrays with large shapes.
             # See https://github.com/numpy/numpy/issues/15753
             lambda shape: prod([i for i in shape if i]) < 100000)

_short_shapes = tuples(integers(0, 10)).filter(
             # numpy gives errors with empty arrays with large shapes.
             # See https://github.com/numpy/numpy/issues/15753
             lambda shape: prod([i for i in shape if i]) < 1000)

# We need to make sure shapes for boolean arrays are generated in a way that
# makes them related to the test array shape. Otherwise, it will be very
# difficult for the boolean array index to match along the test array, which
# means we won't test any behavior other than IndexError.

# short_shapes should be used in place of shapes in any test function that
# uses ndindices, boolean_arrays, or tuples
short_shapes = shared(_short_shapes)

_integer_arrays = arrays(intp, short_shapes)
integer_scalars = arrays(intp, ()).map(lambda x: x[()])
integer_arrays = one_of(integer_scalars, _integer_arrays.flatmap(lambda x: one_of(just(x), just(x.tolist()))))

@composite
def subsequences(draw, sequence):
    seq = draw(sequence)
    start = draw(integers(0, max(0, len(seq)-1)))
    stop = draw(integers(start, len(seq)))
    return seq[start:stop]

_boolean_arrays = arrays(bool_, one_of(subsequences(short_shapes), short_shapes))
boolean_scalars = arrays(bool_, ()).map(lambda x: x[()])
boolean_arrays = one_of(boolean_scalars, _boolean_arrays.flatmap(lambda x: one_of(just(x), just(x.tolist()))))

def _doesnt_raise(idx):
    try:
        ndindex(idx)
    except (IndexError, ValueError, NotImplementedError):
        return False
    return True

Tuples = tuples(one_of(ellipses(), ints(), slices(), newaxes(),
                       integer_arrays, boolean_arrays)).filter(_doesnt_raise)

ndindices = one_of(
    ints(),
    slices(),
    ellipses(),
    newaxes(),
    Tuples,
    integer_arrays,
    boolean_arrays,
).filter(_doesnt_raise)

def assert_equal(actual, desired, err_msg='', verbose=True):
    """
    Same as numpy.testing.assert_equal except it also requires the shapes and
    dtypes to be equal.

    """
    numpy.testing.assert_equal(actual, desired, err_msg=err_msg,
                               verbose=verbose)
    assert actual.shape == desired.shape, err_msg or f"{actual.shape} != {desired.shape}"
    assert actual.dtype == desired.dtype, err_msg or f"{actual.dtype} != {desired.dtype}"

def check_same(a, idx, raw_func=lambda a, idx: a[idx],
               ndindex_func=lambda a, index: a[index.raw],
               same_exception=True, assert_equal=assert_equal):
    """
    Check that a raw index idx produces the same result on an array a before
    and after being transformed by ndindex.

    Tests that raw_func(a, idx) == ndindex_func(a, ndindex(idx)) or that they
    raise the same exception. If same_exception=False, it will still check
    that they both raise an exception, but will not require the exception type
    and message to be the same.

    By default, raw_func(a, idx) is a[idx] and ndindex_func(a, index) is
    a[index.raw].

    The assert_equal argument changes the function used to test equality. By
    default it is the custom assert_equal() function in this file that extends
    numpy.testing.assert_equal. If the func functions return something other
    than arrays, assert_equal should be set to something else, like

        def assert_equal(x, y):
            assert x == y

    """
    exception = None
    try:
        # Handle list indices that NumPy treats as tuple indices with a
        # deprecation warning. We want to test against the post-deprecation
        # behavior.
        e_inner = None
        try:
            try:
                a_raw = raw_func(a, idx)
            except Warning as w:
                if ("Using a non-tuple sequence for multidimensional indexing is deprecated" in w.args[0]):
                    idx = array(idx)
                    a_raw = raw_func(a, idx)
                elif "Out of bound index found. This was previously ignored when the indexing result contained no elements. In the future the index error will be raised. This error occurs either due to an empty slice, or if an array has zero elements even before indexing." in w.args[0]:
                    same_exception = False
                    raise IndexError
                else: # pragma: no cover
                    fail(f"Unexpected warning raised: {w}")
        except Exception:
            _, e_inner, _ = sys.exc_info()
        if e_inner:
            raise e_inner
    except Exception as e:
        exception = e

    try:
        index = ndindex(idx)
        a_ndindex = ndindex_func(a, index)
    except Exception as e:
        if not exception:
            fail(f"Raw form does not raise but ndindex form does ({e!r}): {index})") # pragma: no cover
        if same_exception:
            assert type(e) == type(exception), (e, exception)
            assert e.args == exception.args, (e.args, exception.args)
    else:
        if exception:
            fail(f"ndindex form did not raise but raw form does ({exception!r}): {index})") # pragma: no cover

    if not exception:
        assert_equal(a_raw, a_ndindex)


def iterslice(start_range=(-10, 10),
               stop_range=(-10, 10),
               step_range=(-10, 10),
               one_two_args=True
):
    # one_two_args is unnecessary if the args are being passed to slice(),
    # since slice() already canonicalizes missing arguments to None. We do it
    # for Slice to test that behavior.
    if one_two_args:
        for start in chain(range(*start_range), [None]):
            yield (start,)

        for start in chain(range(*start_range), [None]):
            for stop in chain(range(*stop_range), [None]):
                yield (start, stop)

    for start in chain(range(*start_range), [None]):
        for stop in chain(range(*stop_range), [None]):
            for step in chain(range(*step_range), [None]):
                yield (start, stop, step)


chunk_shapes = shared(shapes)

@composite
def chunk_sizes(draw, shapes=chunk_shapes):
    shape = draw(shapes)
    return draw(tuples(integers(1, 10), min_size=len(shape),
                       max_size=len(shape)).filter(lambda shape: prod(shape) < 10000))
