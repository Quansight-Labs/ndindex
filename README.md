# ndindex

A Python library for manipulating indices of ndarrays.

ndindex is a library that allows representing and manipulating objects that
can be valid indices to numpy arrays, i.e., slices, integers, ellipses,
None, integer and boolean arrays, and tuples thereof. The goals of the library
are

- Provide a uniform API to manipulate these objects. Unlike the standard index
  objects themselves like `slice`, `int`, and `tuple`, which do not share any
  methods in common related to being indices, ndindex classes can all be
  manipulated uniformly. For example, `idx.args` always gives the arguments
  used to construct `idx`.

- Give 100% correct semantics as defined by numpy's ndarray. This means that
  ndindex will not make a transformation on an index object unless it is
  correct for all possible input array shapes. The only exception to this rule
  is that ndindex assumes that any given index will not raise IndexError (for
  instance, from an out of bounds integer index or from too few dimensions).
  For those operations where the array shape is known, there is a `reduce`
  method to reduce an index to a simpler index that is equivalent for the
  given shape.

- Enable useful transformation and manipulation functions on index objects.

## Motivation

If you've ever worked with Python's `slice` objects, you will quickly discover
their limitations:

- Extracting the arguments of a `slice` is cumbersome. You have to write
  `start, stop, step = s.start, s.stop, s.step`. With ndindex you can write
  `start, stop, step = s.args`

- `slice` objects are not hashable. If you want to use them as dictionary
  keys, you have to use cumbersome translation back and forth to a hashable
  type such as `tuple`.

- `slice` makes no assumptions about what they are slicing. As a result,
  invalid slices like `slice(0.5)` or `slice(0, 10, 0)` are allowed. Also
  slices that would always be equivalent like `slice(None, 10)` and `slice(0,
  10)` are unequal. To contrast, ndindex objects always assume they are
  indices to numpy arrays and type check their input. The `reduce` method can
  be used to put the arguments into canonical form.

- Once you generalizing `slice` objects to more general indices, it is
  difficult to work with them in a uniform way. For example, `a[i]` and
  `a[(i,)]` are always equivalent for numpy arrays, but `tuple`, `slice`,
  `int`, etc. are not related to one another. To contrast, all ndindex types
  have a uniform API, and all relevant operations on them produce ndindex
  objects.

- The above limitations can be annoying, but you might consider them worth
  living with. The real pain comes when you start trying to do slice
  arithmetic. Slices in Python behave fundamentally differently depending on
  whether the step is positive or negative and the start and stop are
  positive, negative, or None. Consider, for example, the meaning of the slice
  `a[4:-2:-2]`, where `a` is a one-dimensional array. This slices every other
  element from the third element to the second from the last. The resulting
  array will have shape `(0,)` if the original shape is less than 1 or greater
  than 5, and shape `(1,)` otherwise.

  ndindex pre-codes common slice arithmetic into useful abstractions so you
  don't have to try to figure out all the different cases yourself. And due to
  extensive testing (see below), you can be assured that ndindex is correct.

## Features

ndindex is still a work in progress. The following things are currently
implemented:

- `Slice`, `Integer`, and `Tuple`

- Classes do not canonicalize by default (the constructor only does basic type
  checking). Objects can be put into canonical form by calling `reduce()`.

      >>> from ndindex import Slice
      >>> Slice(None, 12)
      Slice(None, 12, None)
      >>> Slice(None, 12).reduce()
      Slice(0, 12, 1)

- Object arguments can be accessed with `idx.args`

      >>> Slice(1, 3).args
      (1, 3, None)

- All ndindex objects are hashable and can be used as dictionary keys.

- A real index object can be accessed with `idx.raw`. Use this to use an
  ndindex to index an array.

      >>> s = Slice(0, 2)
      >>> from numpy import arange
      >>> arange(4)[s.raw]
      array([0, 1])

- `len()` computes the maximum length of an index over a given axis.

      >>> len(Slice(2, 10, 3))
      3
      >>> len(arange(10)[2:10:3])
      3

- `idx.reduce(shape)` reduces an index to an equivalent index over an array
  with the given shape.

      >>> Slice(2, -1).reduce((10,))
      Slice(2, 9, 1)
      >>> arange(10)[2:-1]
      array([2, 3, 4, 5, 6, 7, 8])
      >>> arange(10)[2:9:1]
      array([2, 3, 4, 5, 6, 7, 8])


The following things are not yet implemented, but are planned.

- `idx.newshape(shape)` returns the shape of `a[idx]`, assuming `a` has shape
  `shape`.

- `ellipsis`, `Newaxis`, `IntegerArray`, and `BooleanArray` types, so that all
  types of indexing are support.

- `i1[i2]` will create a new ndindex `i3` (when possible) so that
  `a[i1][i2] == a[i3]`.

- `split(i0, [i1, i2, ...])` will return a list of indices `[j1, j2, ...]`
  such that `a[i0] = concat(a[i1][j1], a[i2][j2], ...)`

- `i1 + i2` will produce a single index so that `a[i1 + i2]` gives all the
  elements of `a[i1]` and `a[i2]`.

- Support [NEP 21 advanced
  indexing](https://numpy.org/neps/nep-0021-advanced-indexing.html).

And more. If there is something you would like to see this library be able to
do, please [open an issue](https://github.com/quansight/ndindex/issues). Pull
requests are welcome as well.

## Testing and correctness

The most important priority for a library like this is correctness. Index
manipulations, and especially slice manipulations, are complicated to code
correctly, and the code for them typically involves dozens of different
branches for different cases.

In order to assure correctness, all operations are tested extensively against
numpy itself to ensure they give the same results. The basic idea is to take
the pure Python `index` and the `ndindex(index).raw`, or in the case of a
transformation, the before and after raw index, and index a `numpy.arange`
with them (the input array itself doesn't matter, so long as its values are
distinct). If they do not give the same output array, or do not both produce
the same error (like an `IndexError`), the code is not correct. For example,
the `reduce` method can be verified by checking that `a[idx.raw]` and
`a[idx.reduce(a.shape).raw]` produce the same sub-arrays for all possible
input arrays `a` and ndindex objects `idx`.

There are two primary types of tests that we employ to verify this:

- Exhaustive tests. These test every possible value in some range. For
  example, slice tests test all possible `start`, `stop`, and `step` values in
  the range [-10, 10], as well as `None`, on `numpy.arange(n)` for `n` in the
  range [0, 10]. This is the best type of test, because it checks every
  possible case. Unfortunately, it is often impossible to do full exhaustive
  testing due to combinatorial explosion.

- Hypothesis tests. Hypothesis is a library that can intelligently check a
  combinatorial search space of inputs. This requires writing hypothesis
  strategies that can generate all the relevant types of indices (see
  ndindex/tests/helpers.py). For more information on hypothesis, see
  https://hypothesis.readthedocs.io/en/latest/index.html. All tests have
  hypothesis tests, even if they are also tested exhaustively.

Why bother with hypothesis if the same thing is already tested exhaustively?
The main reason is that hypothesis is much better at producing human-readable
failure examples. When an exhaustive test fails, the failure will always be
from the first set of inputs in the loop that produces a failure. Hypothesis
on the other hand attempts to "shrink" the failure input to smallest input
that still fails. For example, a failing exhaustive slice test might give
`Slice(-10, -9, -10)` as a the failing example, but hypothesis would shrink it
to `Slice(-2, -1, -1)`. Another reason for the duplication is that hypothesis
can sometimes test a slightly expanded test space without any additional
consequences. For example, `test_slice_reduce_hypothesis()` in
ndindex/tests/test_ndindex.py tests all types of array shapes, whereas
`test_slice_reduce_exhaustive()` tests only 1-dimensional shapes. This doesn't
affect things because hypotheses will always shrink large shapes to a
1-dimensional shape in the case of a failure. Consequently every exhaustive
test will also have a corresponding hypothesis test.

## License

MIT License

## Table of Contents

* [ndindex Docs Main Page](index.md)
* [ndindex API](api.md)
* [Changelog](changelog.md)
