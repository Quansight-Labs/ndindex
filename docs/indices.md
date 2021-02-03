# Indices

This section of the ndindex documentation discusses the semantics of NumPy
indices. This is more documentation for NumPy itself than for ndindex (and for
basic single dimension indices, the built-in Python sequence types such as
`list` and `str`). However, understanding the underlying semantics of indices
is critical making the best use of ndindex, as well as for making the best use
of NumPy arrays themselves.

There are 7 types of indices supported by NumPy, which correspond to the [8
top-level types](index-types). These can be sorted into three categories:

### Basic single-axis indices

These are the indices that only work on a single axis of an array at a time.
These indices also work for built-in sequence types such as `list` and `str`,
and use the exact same semantics for which elements they select.

- [Integer indices](integer-indices), corresponding to
  [`ndindex.Integer`](integer-api).
- [Slices](slices-docs), corresponding to [`ndindex.Slice`](slice-api).

### Basic multi-axis indices

These are the indices that operate on multiple axes at once. These indices
will not work on the built-in Python sequence types like `list` and `str`;
they are only defined for NumPy arrays. However, like the basic single-axis
indices, these indices are "basic indices", meaning that it returns a
[view](https://numpy.org/doc/stable/glossary.html#term-view) of an array.

- [tuples], corresponding to [`ndindex.Tuple`](tuple-api).
- [ellipses], corresponding to [`ndindex.ellipsis`](ellipsis-api)
- [newaxes] (i.e., `None`), corresponding to [`ndindex.Newaxis`](newaxis-api).

### Advanced indices

Advanced indices operate in general on multiple axes at once. However, unlike
the basic indices, advanced indices in NumPy always return a copy of the array.

- [integer arrays], corresponding to
  [`ndindex.IntegerArray`](integerarray-api).
- [boolean arrays], corresponding to [`ndindex.BooleanArray`](booleanarray-api).

## What is an index?

Nominally, an index is any object that can go between the square brackets
after an array. That is, if `a` is a NumPy array, then in `a[x]`, *`x`* is an
*index* of `a`. This also applies to built-in sequence types in Python such as
`list`,`tuple`, and `str`, but be careful to not confuse the same notation
used on Python dictionaries. If `d` is a Python dictionary, it uses the same
notation `d[x]`, but the meaning of `x` is completely different than what is
being discussed in this document (and indeed, many index types will not even
work if you try them on a dictionary). This document also does not apply to
indexing on Pandas DataFrame or Series objects, except insomuch as they reuse
the same semantics as NumPy.

Semantically, an index `x` picks some subset of the elements of `a`. An index
`a[x]` always either returns a new array with the same dtype as `a`, and
indeed, some subset of the same elements that were in `a`, or it raises
`IndexError`.

**Critically, indices do not in any way depend on the *values* of the
elements they select. They only depend on their *position* in the array `a`.**

For example, suppose `a` is an array of integers of shape `(2, 3, 2)`:

```py
>>> import numpy as np
>>> a = np.array([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
>>> a.shape
(2, 3, 2)
```

Let's take as an example, the index `0, ..., 1:`. We'll investigate how
exactly the index works later. For now, just notice that `a[0, ..., 1:]`
returns a new array with some of the elements of `a`.

```py
>>> a[0, ..., 1:]
array([[1],
       [3],
       [5]])
```

If instead we had an array `b` with the exact same shape `(2, 3, 2)`, but
completely different entries, say, strings, we would find that the index `(0,
..., 1:)` chooses the exact same corresponding elements.

```py
>>> b = np.array([[['A', 'B'], ['C', 'D'], ['E', 'F']], [['G', 'H'], ['I', 'J'], ['K', 'L']]])
>>> b[0, ..., 1:]
array([['B'],
       ['D'],
       ['F']], dtype='<U1')
```

Notice that `'B'` is in the same place in `b` as `1` was in `a`, `'D'` as `3`,
and `'F'` as `5`. Furthermore, the shapes of the resulting arrays are the
same:

```py
>>> a[0, ..., 1:].shape
(3, 1)
>>> b[0, ..., 1:].shape
(3, 1)
```

So the following are always true about any index:

- An index on an array always produces a new array (unless it raises
  `IndexError`).
- The elements of the new array correspond to elements of the original array.
- These elements are chosen by their position in the original array only.
  Their value is irrelevant.
- As such, the exact same index on any other array with the same shape
  produces an array with the exact same corresponding elements.

To be sure, it is possible to *construct* indices that chose specific elements
based on their values. A common example of this is masks (i.e., [boolean array
indices]), such as `a[a > 0]`. However, the resulting index *itself* does not
depend on values (`a > 0` is simply an array of booleans). It can be reused
for any other array with the same shape and it would select exactly the same
elements.

The full range of valid indices allow generating more or less arbitrary new
arrays whose elements come from the indexed array `a`. In practice, the most
commonly desired indices are represented by the basic indices such as integer
indices, slices, and ellipses.