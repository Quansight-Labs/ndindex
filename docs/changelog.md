# ndindex Changelog

## Version 1.2 (2020-05-01)

### Major Changes

- Added `ellipsis` to represent ellipsis indices (`...`). See
  [ellipsis](api.html#ellipsis).

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
