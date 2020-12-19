# ndindex Changelog

## Version 1.5 (2020-12-18)

### Major Changes

- ndindex has been moved to the
  [Quansight-Labs](https://github.com/quansight-labs) organization on Github.
  ndindex is now a [Quansight Labs](https://labs.quansight.org/) project.

- Python 3.6 support has been dropped. ndindex has been tested with Python
  3.7-3.9.

- [`Slice.reduce()`](Slice.reduce) now gives a fully canonical result, meaning
  that two slices `s1` and `s2` are equal on all array shapes if and only if
  `s1.reduce() == s2.reduce()`, and are equal on an array of shape `shape` if
  and only if `s1.reduce(shape) == s2.reduce(shape)` (note that `s1 == s2`
  is only True if `s1` and `s2` are exactly equal). See the
  [documentation](Slice.reduce) for more information on what properties are
  true after canonicalization.

- Make `np.ndarray == ndindex(array)` give `True` or `False` (@telamonian).

- Add [ChunkSize](`ChunkSize`), a new object to represent chunking over an
  array and manipulate indices over chunks.

### Minor Changes

- Various performance improvements.

- Make `hash(idx) == hash(idx.raw)` whenever `idx.raw` is hashable.

- Fix the background color for some monospace text in the docs.

- Fix math formatting in the slices documentation.

## Version 1.4 (2020-09-14)

### Major Changes

- New object [`Newaxis`](Newaxis) to represent `np.newaxis` (i.e., `None`).

- New object [`BooleanArray`](BooleanArray) to represent boolean array indices
  (i.e., masks).

- New object [`IntegerArray`](IntegerArray) to represent integer array indices
  (i.e., fancy indexing).

With these three new objects, ndindex can now represent all valid NumPy index
types. However, note that two corner cases with tuples of arrays are not
implemented:

  - separating arrays by slices, ellipsis, or newaxis (e.g., `a[[0], :, [0]]`)
  - mixing boolean scalars (`True` or `False`) with other non-boolean scalar
    arrays (e.g., `a[True, [0]]`)

The first case in particular may not ever be implemented, as it is considered
to be a mistake that it is allowed in the first place in the NumPy, so if you
need it, please [let me know](https://github.com/Quansight-Labs/ndindex/issues).

Additionally, some corner cases of array semantics are either deprecated or
fixed as bugs in newer versions of NumPy, with some only being fixed in the
unreleased NumPy 1.20. ndindex follows the NumPy 1.20 behavior, and whenever
something is deprecated, ndindex follows the post-deprecation semantics. For
example, in some cases in NumPy, using a list as an index results in it being
interpreted as a tuple and a deprecation warning raised. ndindex always treats
lists as arrays. As another example, there are some cases involving integer
array indices where NumPy does not check bounds but raises a deprecation
warning, and in these cases ndindex does check bounds (when relevant, e.g., in
[`idx.reduce(shape)`](Tuple.reduce)). ndindex should work just fine with older
versions of NumPy, but at least 1.20 (the development version) is required to
run the ndindex test suite due to the way ndindex tests itself against NumPy.

- New method [`broadcast_arrays()`](NDIndex.broadcast_arrays). This will
  convert all boolean arrays into the equivalent integer arrays and broadcast
  all arrays in a `Tuple` together so that they have the same shape.
  `idx.broadcast_arrays()` is equivalent to `idx` in all cases where `idx`
  does not give `IndexError`. Note that broadcastability itself is checked in
  the `Tuple` constructor, so if you only wish to check if an index is valid,
  it is not necessary to call `broadcast_arrays()`.

- [`expand()`](NDIndex.expand) now broadcasts all array inputs (same as
  `broadcast_arrays()`), and combines multiple scalar booleans.

- [`Tuple.reduce()`](Tuple.reduce) now combines multiple scalar booleans.

- [`as_subindex()`](NDIndex.as_subindex) now supports many cases involving
  `BooleanArray` and `IntegerArray`. There are still many instances where
  `as_subindex()` raises `NotImplementedError` however. If you need support
  for these, please [open an
  issue](https://github.com/Quansight-Labs/ndindex/issues) to let me know.

- Add a new document to the documentation on [type confusion](type-confusion).
  The document stresses that ndindex types should not be confused with the
  built-in/NumPy types that they wrap, and outlines some pitfalls and best
  practices to avoid them when using ndindex.

### Minor Changes

- There is now only one docstring for [`expand()`](NDIndex.expand), on the
  `NDindex` base class.

- Calling `Tuple` with a `tuple` argument now raises `ValueError`. Previously
  it raised `NotImplementedError`, because NumPy sometimes treats tuples as
  arrays. It was decided to not allow treating a tuple as an array to avoid
  the [type confusion](type-confusion-tuples) between `Tuple((1, 2))` and
  `Tuple(1, 2)` (only the latter form is correct).

- Document the [`.args`](args) attribute.

- New internal function [`operator_index()`](operator_index), which acts like
  `operator.index()` except it disallows boolean types. A consequence of this
  is that calling the [`Integer`](Integer) or [`Slice`](Slice) constructors
  with boolean arguments will now result in a `TypeError`. Note that scalar
  booleans (`False` and `True`) are valid indices, but they are not the same
  as the integer indices `0` and `1`.

## Version 1.3.1 (2020-07-22)

### Major Changes

- `as_subindex` now supports more input types. In particular, `Integer` is now
  supported better.

- `as_subindex` will now raise `ValueError` in some cases when the two indices
  do not intersect with each other. This is because representing the correct
  answer is either impossible or requires an index type that is not yet
  implemented in ndindex.

### Minor Changes

- `as_subindex` correctly gives `NotImplementedError` for Tuples with
  ellipses.

- `ndindex(list/array/bool/None)` now correctly raise `NotImplementedError`
  instead of `TypeError`.

- `ndindex()` now raises `IndexError` instead of `TypeError` on invalid index
  types, with error messages that match NumPy. This also applies to various
  API functions that call `ndindex()` on their arguments.

- Update the "too many indices for array" error messages to match NumPy 1.19.

- Better checking of arguments for functions that take a shape. The allowed
  shapes and exceptions should now match NumPy more closely (although unknown
  (negative) dimensions do not make sense and are not allowed). Shapes with
  NumPy integer types now work properly.

## Version 1.3 (2020-06-29)

### Major Changes

- New method [`expand`](NDIndex.expand), which always returns a Tuple which is
  as explicit as possible.

- New method [`newshape`](NDIndex.newshape), which returns the shape of an
  array of shape `shape` after being indexed by `idx`.

- New method [`as_subindex`](NDIndex.as_subindex) produces an index `k` such
  that `a[j][k]` gives all the elements of `a[j]` that are also in `a[i]` (see
  the [documentation](NDIndex.as_subindex) for more information). This is
  useful for re-indexing an index onto chunks of an array. Note that this
  raises `NotImplementedError` for some index types that are not yet
  implemented.

- New method [`isempty`](NDIndex.isempty) returns True if an index always
  indexes to an empty array (an array with a 0 in its shape). `isempty` can
  also be called with a shape like `idx.isempty(shape)`.

- SymPy is now a hard dependency of ndindex.

- Added extensive documentation on [slice semantics](slices) to the
  documentation.

### Minor Changes

- Made `Slice.reduce()` give a canonical empty slice in some more cases.

- Move the documentation from recommonmark to Myst.

- Add a [documentation style guide](style-guide).

- Add "See Also" sections to various docstrings.

- Replace the "Fork me on GitHub" ribbon in the documentation with a pure CSS
  version.

- Fix a font name typo in the docs CSS.

- Use a custom doctest runner instead of pytest-doctest. Among other things,
  this now ensures all library doctests include all imports required for the
  doctest to run in a fresh interpreter.

- Various improvements to test coverage.

- Various minor improvements to documentation.

## Version 1.2 (2020-05-01)

### Major Changes

- Added `ellipsis` to represent ellipsis indices (`...`). See
  [ellipsis](ellipsis).

### Minor Changes

- Make `str(Tuple)` more readable.

- Fix a bug in `==` when comparing against non-ndindex types.

- Make `==` give `True` when comparing against equivalent non-ndindex types.

- Make `inspect.signature` give the correct thing for ndindex types.

- Fix `Tuple.reduce()` with no arguments.

- ndindex now has 100% test coverage.

## Version 1.1 (2020-04-23)

### Major Changes

- ndindex objects no longer automatically canonicalize on instantiation. To
  canonicalize an index, call `idx.reduce()` with no arguments. This will put
  it in a canonical form that is equivalent for all array shapes (assuming no
  IndexErrors). `idx.reduce(shape)` can be used to further canonicalize an
  index given that it will index an array of shape `shape`.

### Minor Changes

- Added internal classes to the Sphinx documentation.
- Fixed incorrect `Slice.stop` and `Slice.step`.
- Added the LICENSE file to the source distribution (@synapticarbors).

## Version 1.0 (2020-04-08)

### Major Changes

* Initial ndindex release
* Includes support for `Slice`, `Integer`, and `Tuple`
* Implements basic canonicalization, `.args`, hashability, `.raw`, `len()`,
  and `reduce`.
