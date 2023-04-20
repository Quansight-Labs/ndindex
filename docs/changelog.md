# ndindex Changelog

## Version 1.7 (2023-04-20)

## Major Changes

- **Breaking:** the `skip_axes` argument {func}`~.iter_indices` function now
  applies the skipped axes *before* broadcasting, not after. This behavior is
  more generally useful and matches how functions with stacking work (e.g.,
  `np.cross` or `np.matmul`). The best way to get the old behavior is to
  broadcast the arrays/shapes together first. The `skip_axes` in
  `iter_indices` must be either all negative or all nonnegative to avoid
  ambiguity. A future version may add support for specifying different skip
  axes for each shape.

- {func}`~.iter_indices` no longer requires the skipped axes specified by
  `skip_axes` to be broadcast compatible.

- New method {meth}`~.isvalid` to check if an index is valid on a given shape.

- New function {func}`~.broadcast_shapes` which is the same as
  `np.broadcast_shapes()` except it also allows specifying a set of
  `skip_axes` which will be ignored when broadcasting.

- New exceptions {class}`~.BroadcastError` and {class}`~.AxisError` which are
  used by {func}`~.iter_indices` and {func}`~.broadcast_shapes`.

## Minor Changes

- The documentation theme has been changed to
  [Furo](https://pradyunsg.me/furo/), which has a more clean color scheme
  based on the ndindex logo, better navigation and layout, mobile support, and
  dark mode support.

- Fix some test failures with the latest version of NumPy.

- Fix some tests that didn't work properly when run against the sdist.

- The sdist now includes relevant testing files.

- Automatically deploy docs from CI again.

- Add a documentation preview CI job.

- Test Python 3.11 in CI.

- Minor improvements to some documentation.

- Fix a typo in the [type confusion](type-confusion) docs. (@ruancomelli)

## Version 1.6 (2022-01-24)

### Major Changes

- SymPy is no longer a dependency of ndindex.

- NumPy is now an optional dependency of ndindex. It is only required when
  constructing array indices {class}`~.BooleanArray` or
  {class}`~.IntegerArray`. This does not change the semantics of ndindex.
  ndindex objects still match NumPy indexing semantics everywhere. Note that
  NumPy is still a hard requirement for all tests in the ndindex test suite.

- Added a new function {func}`~.iter_indices` which is a generalization of the
  `np.ndindex()` function (which is otherwise unrelated) to allow multiple
  broadcast compatible shapes, and to allow skipping axes.

- Added a new method {any}`ChunkSize.containing_block`, which computes the
  smallest continuous block of chunks containing a given index.

- ndindex can now be installed with optional Cythonization support. This is
  still experimental and is only enabled when installing ndindex from source
  when Cython is installed (see [the installation
  instructions](installation)). This improves the general performance of
  ndindex.

### Minor Changes

- Fix an issue with the {class}`~.Tuple` constructor with broadcast incompatible
  arrays with the latest version of NumPy.

- Small performance improvement to {any}`Tuple.reduce`.

- Add better support for boolean scalar indices in various
  {class}`~.ChunkSize` methods.

- Better `NotImplementedError` messages from {class}`~.ChunkSize` methods.

- Switch from Travis CI to GitHub Actions.

- Update CI to test Python 3.10.

- Remove Codecov from CI.

- The docs now have a cleaner sidebar which always stays fixed on screen.

## Version 1.5.2 (2021-04-06)

### Major Changes

- ndindex now has a logo: ![ndindex logo](_static/ndindex_logo_white_bg.svg)
  Thanks to [Irina Fumarel](mailto:ifumarel@quansight.com) for the logo design.

- Improve {any}`ChunkSize.as_subchunks()` to never use the slow fallback
  method. This in particular improves the performance for array indices.

- Add a new function {any}`ChunkSize.num_subchunks()`. This is a more efficient
  way of computing `len(list(chunk_size.as_subindex(idx, shape)))`.

### Minor Changes

- Added
  [CODE_OF_CONDUCT.md](https://github.com/Quansight-Labs/ndindex/blob/master/CODE_OF_CONDUCT.md)
  to the ndindex repository. ndindex follows the [Quansight Code of
  Conduct](https://github.com/Quansight/.github/blob/master/CODE_OF_CONDUCT.md).

- Avoid precomputing all iterated values for slices with large steps in
  {any}`ChunkSize.as_subchunks()`.

- Improve the performance of {any}`Slice.__len__()` for slices that
  are already reduced.

- Some minor general performance improvements from moving imports outside of
  functions and adding `__slots__` to all classes.

- Add an [acknowledgments section](acknowledgments) to the README and docs.

## Version 1.5.1 (2021-02-03)

### Major Changes

- The [`ChunkSize.as_subchunks`](ChunkSize.as_subchunks) method now only
  iterates the chunk indices. Previously it iterated `(c,
  index.as_subindex(c))`. But the subindex can always be computed manually
  (there was nothing more efficient about the way it was computed previously),
  and this is much slower if the subindex is not actually needed. This is a
  backwards incompatible change, but since the `ChunkSize` object was only
  introduced in the previous release, it should hopefully not have a major
  impact.

- Made improvements to performance throughout the library. The improvements in
  some instances are drastic.

- Added a benchmarking suite using [airspeed
  velocity](https://asv.readthedocs.io/en/stable/). Graphs of the benchmarks
  can be viewed at <https://quansight-labs.github.io/ndindex/benchmarks/>.

## Version 1.5 (2020-12-23)

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

- Add {class}`~.ChunkSize`, a new object to represent chunking over an array
  and manipulate indices over chunks.

### Minor Changes

- Various performance improvements.

- Make `hash(idx) == hash(idx.raw)` whenever `idx.raw` is hashable.

- Fix the background color for some monospace text in the docs.

- Fix math formatting in the slices documentation.

## Version 1.4 (2020-09-14)

### Major Changes

- New object {class}`~.Newaxis` to represent `np.newaxis` (i.e., `None`).

- New object {class}`~.BooleanArray` to represent boolean array indices (i.e.,
  masks).

- New object {class}`~.IntegerArray` to represent integer array indices (i.e.,
  fancy indexing).

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

- Document the [`.args`](ndindex.ndindex.ImmutableObject.args) attribute.

- New internal function {func}`~.operator_index`, which acts like
  `operator.index()` except it disallows boolean types. A consequence of this
  is that calling the {class}`~.Integer` or {class}`~.Slice` constructors with
  boolean arguments will now result in a `TypeError`. Note that scalar
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
  {class}`~.ellipsis`.

### Minor Changes

- Make `str(Tuple)` more readable.

- Fix a bug in `==` when comparing against non-ndindex types.

- Make `==` give `True` when comparing against equivalent non-ndindex types.

- Make `inspect.signature` give the correct thing for ndindex types.

- Fix {any}`Tuple.reduce()` with no arguments.

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
