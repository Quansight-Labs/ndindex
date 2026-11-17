import sys
from itertools import chain
import warnings
from functools import wraps, partial

from numpy import ndarray, generic, intp, bool_, asarray, broadcast_shapes
import numpy.testing

from pytest import fail

from hypothesis import assume, note
from hypothesis.strategies import (integers, none, one_of, lists, just,
                                   builds, shared, composite, sampled_from,
                                   booleans,
                                   nothing, tuples as hypothesis_tuples)
from hypothesis.extra.numpy import (arrays, mutually_broadcastable_shapes as
                                    mbs, BroadcastableShapes, valid_tuple_axes)
from hypothesis.errors import UnsatisfiedAssumption

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

_short_shapes = lambda n: tuples(integers(0, 10), min_size=n, max_size=32).filter(
             # numpy gives errors with empty arrays with large shapes.
             # See https://github.com/numpy/numpy/issues/15753
             lambda shape: prod([i for i in shape if i]) < SHORT_MAX_ARRAY_SIZE)

# short_shapes is the shape of the array that is being indexed. It should be
# used in place of shapes in any test function that uses ndindices,
# integer_arrays, boolean_arrays, or Tuples.
#
# short_shapes is a shared() strategy, meaning every strategy in a given test
# that draws from it gets the same shape. The index strategies below make use
# of this to generate indices that are usually valid for an array of that
# shape. An index that is generated independently of the shape is only valid
# by chance, which is rare for anything but the simplest indices (for example,
# a tuple index is only valid if every one of its arguments is in bounds for
# its corresponding axis and the arguments don't index too many axes). That
# would mean that we wouldn't test much behavior other than IndexError for
# more complicated indices.
#
# The index strategies still sometimes generate indices that are independent
# of the shape, so that every combination of index and shape can be generated,
# including invalid ones. Compatible combinations are just the common case.
short_shapes = shared(_short_shapes(0))

def _usually(common, rare):
    """
    Draw from the strategy `common` most of the time and from `rare` the rest
    of the time.

    Failing examples shrink towards `common`.
    """
    return sampled_from([common]*7 + [rare]).flatmap(lambda strategy: strategy)

# Indices that are generated independently of the shape of the array that is
# being indexed.

_integer_arrays = arrays(intp, short_shapes)
integer_scalars = arrays(intp, ()).map(lambda x: x[()])
_arbitrary_integer_arrays = one_of(integer_scalars, _integer_arrays.flatmap(lambda x: one_of(just(x), just(x.tolist()))))

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

_arbitrary_tuple_args = one_of(ellipses(), ints(), slices(), newaxes(),
                               _arbitrary_integer_arrays, boolean_arrays)
_arbitrary_tuples = tuples(_arbitrary_tuple_args)

# Indices that are valid for an array with shape short_shapes.

_nonzero_step_slices = slices(step=one_of(none(), ints().filter(lambda i: i != 0)))

@composite
def _ints_for_axis(draw, axis=0, shapes=short_shapes):
    """
    Generate an integer index that is in bounds for the given axis of the
    shape drawn from `shapes`.

    If there is no such index, an arbitrary integer is generated.
    """
    shape = draw(shapes)
    if len(shape) <= axis or shape[axis] == 0:
        return draw(ints())
    return draw(integers(-shape[axis], shape[axis] - 1))

@composite
def _integer_arrays_for_axis(draw, axis=0, shapes=short_shapes):
    """
    Generate an integer array index that is in bounds for the given axis of
    the shape drawn from `shapes`.

    The array either has that same shape or is 0-D, so that every
    integer array generated for a given test is broadcast compatible with the
    others. If there is no such index, an arbitrary integer array is
    generated.
    """
    shape = draw(shapes)
    if len(shape) <= axis:
        return draw(_arbitrary_integer_arrays)

    n = shape[axis]
    if n == 0:
        # Only an empty array can index an axis of size 0. An array with the
        # same shape as the array being indexed is always empty here.
        return draw(arrays(intp, shape))

    array_shape = draw(sampled_from([shape, ()]))
    x = draw(arrays(intp, array_shape, elements=integers(-n, n - 1)))
    if array_shape == () and draw(booleans()):
        return x[()]
    # Empty lists do not keep their shape
    if x.size != 0 and draw(booleans()):
        return x.tolist()
    return x

integer_arrays = _usually(_integer_arrays_for_axis(), _arbitrary_integer_arrays)

@composite
def _tuples_for_shape(draw, shapes=short_shapes):
    """
    Generate a tuple index that is valid for an array with the shape drawn
    from `shapes`.
    """
    shape = draw(shapes)
    ndim = len(shape)

    # The arguments before the ellipsis index the axes [0, a) and the
    # arguments after it index the axes [b, ndim). If there is no ellipsis,
    # the axes [a, ndim) are not indexed. Index every axis about half the
    # time, since otherwise most tuples would only have a few arguments.
    n_skipped = draw(one_of(just(0), integers(0, ndim)))
    has_ellipsis = draw(booleans())
    a = draw(integers(0, ndim - n_skipped)) if has_ellipsis else ndim - n_skipped
    b = a + n_skipped

    # If a tuple has arrays, all the arrays and integers must be in a single
    # contiguous block, which cannot mix boolean scalars with other arrays
    # (see the Tuple constructor). Furthermore, the arrays in the block must
    # all be broadcast compatible, so we only generate a single boolean array.
    # block is the kind of the block so far, or None if it hasn't been started
    # yet. The block is closed once it is followed by a slice, ellipsis, or
    # newaxis.
    has_arrays = draw(booleans())
    block = None
    block_closed = False

    args = []

    def close_block():
        nonlocal block_closed
        if block is not None:
            block_closed = True

    def nonindexing_args():
        # Newaxes and boolean scalars do not index any axes, so they can go
        # anywhere in the tuple.
        nonlocal block
        while True:
            kinds = ['newaxis']
            if (has_arrays and not block_closed
                and block in [None, 'integer', 'boolean scalar']):
                kinds.append('boolean scalar')
            kind = draw(_usually(none(), sampled_from(kinds)))

            if kind == 'newaxis':
                close_block()
                args.append(None)
            elif kind == 'boolean scalar':
                block = 'boolean scalar'
                args.append(draw(boolean_scalars))
            else:
                break

    def index_axes(start, stop):
        nonlocal block
        axis = start
        while axis < stop:
            nonindexing_args()

            n = shape[axis]
            in_block = has_arrays and not block_closed
            kinds = ['slice']
            if n != 0 and (in_block or not has_arrays):
                kinds.append('integer')
            if in_block and block in [None, 'integer', 'integer array']:
                kinds.append('integer array')
            if in_block and block in [None, 'integer']:
                kinds.append('boolean array')
            kind = draw(sampled_from(kinds))

            if kind == 'slice':
                close_block()
                args.append(draw(_nonzero_step_slices))
                axis += 1
            elif kind == 'integer':
                if has_arrays and block is None:
                    block = 'integer'
                args.append(draw(integers(-n, n - 1)))
                axis += 1
            elif kind == 'integer array':
                block = 'integer array'
                args.append(draw(_integer_arrays_for_axis(axis, just(shape))))
                axis += 1
            else:
                block = 'boolean array'
                # A boolean array indexes as many axes as it has dimensions
                n_axes = draw(integers(1, stop - axis))
                x = draw(arrays(bool_, shape[axis:axis + n_axes]))
                # Empty lists do not keep their shape or dtype
                if x.size != 0 and draw(booleans()):
                    x = x.tolist()
                args.append(x)
                axis += n_axes
        nonindexing_args()

    index_axes(0, a)
    if has_ellipsis:
        close_block()
        args.append(...)
        index_axes(b, ndim)

    return tuple(args)

@composite
def _almost_valid_tuples(draw):
    """
    Generate a valid tuple index with some arbitrary arguments inserted into
    it.
    """
    args = list(draw(_tuples_for_shape()))
    for arg in draw(lists(_arbitrary_tuple_args, min_size=1, max_size=2)):
        args.insert(draw(integers(0, len(args))), arg)
    return tuple(args)

def _doesnt_raise(idx):
    try:
        ndindex(idx)
    except (IndexError, ValueError, NotImplementedError):
        return False
    return True

Tuples = _usually(_tuples_for_shape(),
                  one_of(_almost_valid_tuples(), _arbitrary_tuples)).filter(_doesnt_raise)

ndindices = one_of(
    _usually(_ints_for_axis(), ints()),
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

def _fill_shape(draw,
                *,
                result_shape,
                skip_axes,
                skip_axes_values):
    max_n = max([i + 1 if i >= 0 else -i for i in skip_axes], default=0)
    assume(max_n <= len(skip_axes) + len(result_shape))
    dim = draw(integers(min_value=max_n, max_value=len(skip_axes) + len(result_shape)))
    new_shape = ['placeholder']*dim
    for i in skip_axes:
        assume(new_shape[i] is not None) # skip_axes must be unique
        new_shape[i] = None
    j = -1
    for i in range(-1, -dim - 1, -1):
        if new_shape[i] is None:
            new_shape[i] = draw(skip_axes_values)
        else:
            new_shape[i] = draw(sampled_from([result_shape[j], 1]))
            j -= 1
    while new_shape and new_shape[0] == 'placeholder': # pragma: no cover
        # Can happen if positive and negative skip_axes refer to the same
        # entry
        new_shape.pop(0)

    # This will happen if the skip axes are too large
    assume('placeholder' not in new_shape)

    if prod([i for i in new_shape if i]) >= SHORT_MAX_ARRAY_SIZE:
        note(f"Filtering the shape {new_shape} (too many elements)")
        assume(False)

    return tuple(new_shape)

skip_axes_with_broadcasted_shape_type = shared(sampled_from([int, tuple, list]))

@composite
def _mbs_and_skip_axes(
        draw,
        shapes=short_shapes,
        min_shapes=0,
        max_shapes=32,
        skip_axes_type_st=skip_axes_with_broadcasted_shape_type,
        skip_axes_values=integers(0, 20),
        num_skip_axes=None,
):
    """
    mutually_broadcastable_shapes except skip_axes() axes might not be
    broadcastable

    The result_shape will be None in the position of skip_axes.
    """
    skip_axes_type = draw(skip_axes_type_st)
    _result_shape = draw(shapes)
    if _result_shape == ():
        assume(num_skip_axes is None)

    ndim = len(_result_shape)
    num_shapes = draw(integers(min_value=min_shapes, max_value=max_shapes))
    if not num_shapes:
        assume(num_skip_axes is None)
        num_skip_axes = 0
    if not ndim:
        return BroadcastableShapes([()]*num_shapes, ()), ()

    if num_skip_axes is not None:
        min_skip_axes = max_skip_axes = num_skip_axes
    else:
        min_skip_axes = 0
        max_skip_axes = None

    # int and single tuple cases must be limited to N to ensure that they are
    # correct for all shapes
    if skip_axes_type == int:
        assume(num_skip_axes in [None, 1])
        skip_axes = draw(valid_tuple_axes(ndim, min_size=1, max_size=1))[0]
        _skip_axes = [(skip_axes,)]*num_shapes
    elif skip_axes_type == tuple:
        skip_axes = draw(tuples(integers(-ndim, ndim-1), min_size=min_skip_axes,
                               max_size=max_skip_axes, unique=True))
        _skip_axes = [skip_axes]*num_shapes
    elif skip_axes_type == list:
        skip_axes = []
        for i in range(num_shapes):
            skip_axes.append(draw(tuples(integers(-ndim, ndim+1), min_size=min_skip_axes,
                                         max_size=max_skip_axes, unique=True)))
        _skip_axes = skip_axes

    shapes = []
    for i in range(num_shapes):
        shapes.append(_fill_shape(draw, result_shape=_result_shape, skip_axes=_skip_axes[i],
                                  skip_axes_values=skip_axes_values))

    non_skip_shapes = [remove_indices(shape, sk) for shape, sk in
                            zip(shapes, _skip_axes)]
    # Broadcasting the result _fill_shape may produce a shape different from
    # _result_shape because it might not have filled all dimensions, or it
    # might have chosen 1 for a dimension every time. Ideally we would just be
    # using shapes from mutually_broadcastable_shapes, but I don't know how to
    # reverse inject skip axes into shapes in general (see the comment in
    # unremove_indices). So for now, we just use the actual broadcast of the
    # non-skip shapes. Note that we use np.broadcast_shapes here instead of
    # ndindex.broadcast_shapes because test_broadcast_shapes itself uses this
    # strategy.
    broadcasted_shape = broadcast_shapes(*non_skip_shapes)

    return BroadcastableShapes(shapes, broadcasted_shape), skip_axes

mbs_and_skip_axes = shared(_mbs_and_skip_axes())

mutually_broadcastable_shapes_with_skipped_axes = mbs_and_skip_axes.map(
    lambda i: i[0])
skip_axes_st = mbs_and_skip_axes.map(lambda i: i[1])

@composite
def _cross_shapes_and_skip_axes(draw):
    (shapes, _broadcasted_shape), skip_axes = draw(_mbs_and_skip_axes(
        shapes=_short_shapes(2),
        min_shapes=2,
        max_shapes=2,
        num_skip_axes=1,
        # TODO: Test other skip axes types
        skip_axes_type_st=just(list),
        skip_axes_values=just(3),
    ))

    broadcasted_skip_axis = draw(integers(-len(_broadcasted_shape)-1, len(_broadcasted_shape)))
    broadcasted_shape = unremove_indices(_broadcasted_shape,
                                         [broadcasted_skip_axis], val=3)
    skip_axes.append((broadcasted_skip_axis,))

    return BroadcastableShapes(shapes, broadcasted_shape), skip_axes

cross_shapes_and_skip_axes = shared(_cross_shapes_and_skip_axes())
cross_shapes = cross_shapes_and_skip_axes.map(lambda i: i[0])
cross_skip_axes = cross_shapes_and_skip_axes.map(lambda i: i[1])

@composite
def cross_arrays_st(draw):
    broadcastable_shapes = draw(cross_shapes)
    shapes, broadcasted_shape = broadcastable_shapes

    # Sanity check
    assert len(shapes) == 2
    # We need to generate fairly random arrays. Otherwise, if they are too
    # similar to each other, like two arange arrays would be, the cross
    # product will be 0. We also disable the fill feature in arrays() for the
    # same reason, as it would otherwise generate too many vectors that are
    # colinear.
    a = draw(arrays(dtype=int, shape=shapes[0], elements=integers(-100, 100), fill=nothing()))
    b = draw(arrays(dtype=int, shape=shapes[1], elements=integers(-100, 100), fill=nothing()))

    return a, b

@composite
def _matmul_shapes_and_skip_axes(draw):
    (shapes, _broadcasted_shape), skip_axes = draw(_mbs_and_skip_axes(
        shapes=_short_shapes(2),
        min_shapes=2,
        max_shapes=2,
        num_skip_axes=2,
        # TODO: Test other skip axes types
        skip_axes_type_st=just(list),
        skip_axes_values=just(None),
    ))

    broadcasted_skip_axes = draw(hypothesis_tuples(*[
        integers(-len(_broadcasted_shape)-1, len(_broadcasted_shape))
    ]*2))

    try:
        broadcasted_shape = unremove_indices(_broadcasted_shape,
                                             broadcasted_skip_axes)
    except NotImplementedError:
        # TODO: unremove_indices only works with both positive or both negative
        assume(False)
    # Make sure the indices are unique
    assume(len(set(broadcasted_skip_axes)) == len(broadcasted_skip_axes))

    skip_axes.append(broadcasted_skip_axes)

    # (n, m) @ (m, k) -> (n, k)
    n, m, k = draw(hypothesis_tuples(integers(0, 10), integers(0, 10),
                                     integers(0, 10)))
    shape1, shape2 = map(list, shapes)
    ax1, ax2 = skip_axes[0]
    shape1[ax1] = n
    shape1[ax2] = m
    ax1, ax2 = skip_axes[1]
    shape2[ax1] = m
    shape2[ax2] = k
    broadcasted_shape = list(broadcasted_shape)
    ax1, ax2 = skip_axes[2]
    broadcasted_shape[ax1] = n
    broadcasted_shape[ax2] = k

    shapes = (tuple(shape1), tuple(shape2))
    broadcasted_shape = tuple(broadcasted_shape)

    return BroadcastableShapes(shapes, broadcasted_shape), skip_axes

matmul_shapes_and_skip_axes = shared(_matmul_shapes_and_skip_axes())
matmul_shapes = matmul_shapes_and_skip_axes.map(lambda i: i[0])
matmul_skip_axes = matmul_shapes_and_skip_axes.map(lambda i: i[1])

@composite
def matmul_arrays_st(draw):
    broadcastable_shapes = draw(matmul_shapes)
    shapes, broadcasted_shape = broadcastable_shapes

    # Sanity check
    assert len(shapes) == 2
    a = draw(arrays(dtype=int, shape=shapes[0], elements=integers(-100, 100)))
    b = draw(arrays(dtype=int, shape=shapes[1], elements=integers(-100, 100)))

    return a, b

reduce_kwargs = sampled_from([{}, {'negative_int': False}, {'negative_int': True}])

def assert_equal(actual, desired, allow_scalar_0d=False, err_msg='', verbose=True):
    """
    Assert that two objects are equal.

    - If the objects are ndarrays, this is the same as
      numpy.testing.assert_equal except it also requires the shapes and dtypes
      to be equal

    - If the objects are tuples, recursively call assert_equal to support
      tuples of arrays.

    - If allow_scalar_0d=True, scalars will be considered equal to equivalent
      0-D arrays.

    - Require the types of actual and desired to be exactly the same
      (excepting for scalars when allow_scalar_0d=True).

    """
    if not (allow_scalar_0d and (isinstance(actual, generic)
                                 or isinstance(desired, generic))):
        assert type(actual) is type(desired), err_msg or f"{type(actual)} != {type(desired)}"

    if isinstance(actual, (ndarray, generic)):
        numpy.testing.assert_equal(actual, desired, err_msg=err_msg,
                                   verbose=verbose)
        assert actual.shape == desired.shape, err_msg or f"{actual.shape} != {desired.shape}"
        assert actual.dtype == desired.dtype, err_msg or f"{actual.dtype} != {desired.dtype}"
    elif isinstance(actual, tuple):
        assert len(actual) == len(desired), err_msg
        for i, j in zip(actual, desired):
            assert_equal(i, j, err_msg=err_msg, verbose=verbose)
    else:
        assert actual == desired, err_msg

assert_equal_allow_scalar_0d = partial(assert_equal, allow_scalar_0d=True)

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
               conversion_func=ndindex,
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
            except IndexError as e:
                # It's not straightforward to avoid indexing too many
                # dimensions in the strategy generation, because the total
                # number of dimensions in the result array is not a trivial
                # thing. Furthermore, some versions of NumPy limit this to 32
                # and some limit it to 64.
                if "number of dimensions must be within" in str(e): # pragma: no cover
                    raise UnsatisfiedAssumption
                raise
            except Warning as w:
                # In NumPy < 1.23, this is a FutureWarning. In 1.23 the
                # deprecation was removed and lists are always interpreted as
                # array indices.
                if ("Using a non-tuple sequence for multidimensional indexing is deprecated" in w.args[0]): # pragma: no cover
                    idx = asarray(idx, dtype=intp)
                    a_raw = raw_func(a, idx)
                elif "Out of bound index found. This was previously ignored when the indexing result contained no elements. In the future the index error will be raised. This error occurs either due to an empty slice, or if an array has zero elements even before indexing." in w.args[0]: # pragma: no cover
                    same_exception = False
                    raise IndexError
                else: # pragma: no cover
                    fail(f"Unexpected warning raised: {w}")
        except Exception:
            _, e_inner, _ = sys.exc_info()
        if e_inner:
            raise e_inner
    except UnsatisfiedAssumption: # pragma: no cover
        raise
    except Exception as e:
        exception = e

    index = '<conversion to ndindex object failed>'
    try:
        index = conversion_func(idx)
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


chunk_shapes = short_shapes

@composite
def chunk_sizes(draw, shapes=chunk_shapes):
    shape = draw(shapes)
    return draw(tuples(integers(1, 10), min_size=len(shape),
                       max_size=len(shape)).filter(lambda shape: prod(shape) < MAX_ARRAY_SIZE))
