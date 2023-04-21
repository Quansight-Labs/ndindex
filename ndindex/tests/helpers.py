import sys
from itertools import chain
import warnings
from functools import wraps

from numpy import intp, bool_, array, broadcast_shapes
import numpy.testing

from pytest import fail

from hypothesis import assume, note
from hypothesis.strategies import (integers, none, one_of, lists, just,
                                   builds, shared, composite, sampled_from,
                                   booleans)
from hypothesis.extra.numpy import (arrays, mutually_broadcastable_shapes as
                                    mbs, BroadcastableShapes)

from ..ndindex import ndindex
from ..shapetools import remove_indices, unremove_indices
from .._crt import prod

# Hypothesis strategies for generating indices. Note that some of these
# strategies are nominally already defined in hypothesis, but we redefine them
# here because the hypothesis definitions are too restrictive. For example,
# hypothesis's slices strategy does not generate slices with negative indices.
# Similarly, hypothesis.extra.numpy.basic_indices only generates tuples.

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

MAX_ARRAY_SIZE = 100000
SHORT_MAX_ARRAY_SIZE = 1000
shapes = tuples(integers(0, 10)).filter(
             # numpy gives errors with empty arrays with large shapes.
             # See https://github.com/numpy/numpy/issues/15753
             lambda shape: prod([i for i in shape if i]) < MAX_ARRAY_SIZE)

_short_shapes = lambda n: tuples(integers(0, 10), min_size=n).filter(
             # numpy gives errors with empty arrays with large shapes.
             # See https://github.com/numpy/numpy/issues/15753
             lambda shape: prod([i for i in shape if i]) < SHORT_MAX_ARRAY_SIZE)

# short_shapes should be used in place of shapes in any test function that
# uses ndindices, boolean_arrays, or tuples
short_shapes = shared(_short_shapes(0))

_integer_arrays = arrays(intp, short_shapes)
integer_scalars = arrays(intp, ()).map(lambda x: x[()])
integer_arrays = one_of(integer_scalars, _integer_arrays.flatmap(lambda x: one_of(just(x), just(x.tolist()))))

# We need to make sure shapes for boolean arrays are generated in a way that
# makes them related to the test array shape. Otherwise, it will be very
# difficult for the boolean array index to match along the test array, which
# means we won't test any behavior other than IndexError.

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

# Note: We could use something like this:

# mutually_broadcastable_shapes = shared(integers(1, 32).flatmap(lambda i: mbs(num_shapes=i).filter(
#     lambda broadcastable_shapes: prod([i for i in broadcastable_shapes.result_shape if i]) < MAX_ARRAY_SIZE)))

@composite
def _mutually_broadcastable_shapes(draw, *, shapes=short_shapes, min_shapes=0, max_shapes=32, min_side=0):
    # mutually_broadcastable_shapes() with the default inputs doesn't generate
    # very interesting examples (see
    # https://github.com/HypothesisWorks/hypothesis/issues/3170). It's very
    # difficult to get it to do so by tweaking the max_* parameters, because
    # making them too big leads to generating too large shapes and filtering
    # too much. So instead, we trick it into generating more interesting
    # examples by telling it to create shapes that broadcast against some base
    # shape.

    # Unfortunately, this, along with the filtering below, has a downside that
    # it tends to generate a result shape of () more often than you might
    # like. But it generates enough "real" interesting shapes that both of
    # these workarounds are worth doing (plus I don't know if any other better
    # way of handling the situation).
    base_shape = draw(shapes)

    input_shapes, result_shape = draw(
        mbs(
            num_shapes=max_shapes,
            base_shape=base_shape,
            min_side=min_side,
        ))

    # The hypothesis mutually_broadcastable_shapes doesn't allow num_shapes to
    # be a strategy. It's tempting to do something like num_shapes =
    # draw(integers(min_shapes, max_shapes)), but this shrinks poorly. See
    # https://github.com/HypothesisWorks/hypothesis/issues/3151. So instead of
    # using a strategy to draw the number of shapes, we just generate max_shapes
    # shapes and pick a subset of them.
    final_input_shapes = draw(lists(sampled_from(input_shapes),
                                    min_size=min_shapes, max_size=max_shapes))


    # Note: result_shape is input_shapes broadcasted with base_shape, but
    # base_shape itself is not part of input_shapes. We "really" want our base
    # shape to be (). We are only using it here to trick
    # mutually_broadcastable_shapes into giving more interesting examples.
    final_result_shape = broadcast_shapes(*final_input_shapes)

    # The broadcast compatible shapes can be bigger than the base shape. This
    # is already somewhat limited by the mutually_broadcastable_shapes
    # defaults, and pretty unlikely, but we filter again here just to be safe.
    if not prod([i for i in final_result_shape if i]) < SHORT_MAX_ARRAY_SIZE: # pragma: no cover
        note(f"Filtering the shape {result_shape} (too many elements)")
        assume(False)

    return BroadcastableShapes(final_input_shapes, final_result_shape)

mutually_broadcastable_shapes = shared(_mutually_broadcastable_shapes())

@composite
def _skip_axes_st(draw,
                  mutually_broadcastable_shapes=mutually_broadcastable_shapes,
                  num_skip_axes=None):
    shapes, result_shape = draw(mutually_broadcastable_shapes)
    if result_shape == ():
        assume(num_skip_axes is None)
        return ()
    negative = draw(booleans(), label='skip_axes < 0')
    N = len(min(shapes, key=len))
    if num_skip_axes is not None:
        min_size = max_size = num_skip_axes
        assume(N >= num_skip_axes)
    else:
        min_size = 0
        max_size = None
    if N == 0:
        return ()
    if negative:
        axes = draw(lists(integers(-N, -1), min_size=min_size, max_size=max_size, unique=True))
    else:
        axes = draw(lists(integers(0, N-1), min_size=min_size, max_size=max_size, unique=True))
    axes = tuple(axes)
    # Sometimes return an integer
    if num_skip_axes is None and len(axes) == 1 and draw(booleans(), label='skip_axes integer'): # pragma: no cover
        return axes[0]
    return axes

skip_axes_st = shared(_skip_axes_st())

@composite
def mutually_broadcastable_shapes_with_skipped_axes(draw, skip_axes_st=skip_axes_st, mutually_broadcastable_shapes=mutually_broadcastable_shapes,
skip_axes_values=integers(0)):
    """
    mutually_broadcastable_shapes except skip_axes() axes might not be
    broadcastable

    The result_shape will be None in the position of skip_axes.
    """
    skip_axes_ = draw(skip_axes_st)
    shapes, result_shape = draw(mutually_broadcastable_shapes)
    if isinstance(skip_axes_, int):
        skip_axes_ = (skip_axes_,)

    # Randomize the shape values in the skipped axes
    shapes_ = []
    for shape in shapes:
        _shape = list(unremove_indices(shape, skip_axes_))
        # sanity check
        assert remove_indices(_shape, skip_axes_) == shape, (_shape, skip_axes_, shape)

        # Replace None values with random values
        for j in range(len(_shape)):
            if _shape[j] is None:
                _shape[j] = draw(skip_axes_values)
        shapes_.append(tuple(_shape))

    result_shape_ = unremove_indices(result_shape, skip_axes_)
    # sanity check
    assert remove_indices(result_shape_, skip_axes_) == result_shape

    for shape in shapes_:
        if prod([i for i in shape if i]) >= SHORT_MAX_ARRAY_SIZE:
            note(f"Filtering the shape {shape} (too many elements)")
            assume(False)
    return BroadcastableShapes(shapes_, result_shape_)

two_mutually_broadcastable_shapes_1 = shared(_mutually_broadcastable_shapes(
    shapes=_short_shapes(1),
    min_shapes=2,
    max_shapes=2,
    min_side=1))
one_skip_axes = shared(_skip_axes_st(
    mutually_broadcastable_shapes=two_mutually_broadcastable_shapes_1,
    num_skip_axes=1))
two_mutually_broadcastable_shapes_2 = shared(_mutually_broadcastable_shapes(
    shapes=_short_shapes(2),
    min_shapes=2,
    max_shapes=2,
    min_side=2))
two_skip_axes = shared(_skip_axes_st(
    mutually_broadcastable_shapes=two_mutually_broadcastable_shapes_2,
    num_skip_axes=2))

def assert_equal(actual, desired, err_msg='', verbose=True):
    """
    Same as numpy.testing.assert_equal except it also requires the shapes and
    dtypes to be equal.

    """
    numpy.testing.assert_equal(actual, desired, err_msg=err_msg,
                               verbose=verbose)
    assert actual.shape == desired.shape, err_msg or f"{actual.shape} != {desired.shape}"
    assert actual.dtype == desired.dtype, err_msg or f"{actual.dtype} != {desired.dtype}"

def warnings_are_errors(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return f(*args, **kwargs)
    return inner

@warnings_are_errors
def check_same(a, idx, *, raw_func=lambda a, idx: a[idx],
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
                # In NumPy < 1.23, this is a FutureWarning. In 1.23 the
                # deprecation was removed and lists are always interpreted as
                # array indices.
                if ("Using a non-tuple sequence for multidimensional indexing is deprecated" in w.args[0]): # pragma: no cover
                    idx = array(idx, dtype=intp)
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
                       max_size=len(shape)).filter(lambda shape: prod(shape) < MAX_ARRAY_SIZE))
