from itertools import chain
from functools import reduce
from operator import mul

from numpy.testing import assert_equal

from pytest import fail

from hypothesis import assume
from hypothesis.strategies import integers, composite, none, one_of, lists, just

from ..ndindex import ndindex

# Hypothesis strategies for generating indices. Note that some of these
# strategies are nominally already defined in hypothesis, but we redefine them
# here because the hypothesis definitions are too restrictive. For example,
# hypothesis's slices strategy does not generate slices with negative indices.
# Similarly, hypothesis.extra.numpy.basic_indices only generates tuples.

# np.prod has overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)

ints = lambda: integers(-10, 10)

@composite
def slices(draw, start=ints(), stop=ints(), step=ints()):
    return slice(
        draw(one_of(none(), start)),
        draw(one_of(none(), stop)),
        draw(one_of(none(), step)),
    )

ellipses = lambda: just(...)

# hypotheses.strategies.tuples only generates tuples of a fixed size
@composite
def tuples(draw, elements, *, min_size=0, max_size=None, unique_by=None,
           unique=False):
    return tuple(draw(lists(elements, min_size=min_size, max_size=max_size,
                            unique_by=unique_by, unique=unique)))

Tuples = tuples(one_of(ellipses(), ints(), slices()))

@composite
def ndindices(draw):
    s = draw(one_of(
        ints(),
        slices(),
        ellipses(),
        tuples(one_of(ints(), slices())),
    ))

    try:
        return ndindex(s)
    except ValueError:
        assume(False)

shapes = tuples(integers(0, 10)).filter(
             # numpy gives errors with empty arrays with large shapes.
             # See https://github.com/numpy/numpy/issues/15753
             lambda shape: prod([i for i in shape if i]) < 100000)

def check_same(a, index, func=lambda x: x, same_exception=True):
    exception = None
    try:
        a_raw = a[index]
    except Exception as e:
        exception = e

    try:
        idx = ndindex(index)
        idx = func(idx)
        a_idx = a[idx.raw]
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
        assert_equal(a_raw, a_idx)



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
