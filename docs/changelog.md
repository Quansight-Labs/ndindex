# ndindex Changelog

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
