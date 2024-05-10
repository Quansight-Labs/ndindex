# Introduction: What is an Index?

Nominally, an index is any object that can be placed between the square
brackets after an array. That is, if `a` is a NumPy array, then in `a[x]`,
*`x`* is an *index* of `a`.[^index-vs-slice-footnote] This also applies to
built-in sequence types in Python, such as `list`, `tuple`, and `str`;
however, be careful not to confuse this with the similar notation used in
Python dictionaries. If `d` is a Python dictionary, it uses the same notation
`d[x]`, but the meaning of `x` is completely different from what is discussed
in this document. This document also does not apply to indexing Pandas
DataFrame or Series objects, except insofar as they reuse the same semantics
as NumPy. Finally, note that some other Python array libraries (e.g., PyTorch
or Jax) have similar indexing rules, but they generally implement only a
subset of the full NumPy semantics outlined here.

[^index-vs-slice-footnote]: Some people call `x` a *slice* of `a`, but we
    avoid this confusing nomenclature, using *slice* to refer only to the
    [slice index type](slices.md). The term "index" is used in the Python
    language itself (e.g., in the built-in exception type `IndexError`).

Semantically, an index `x` selects, or *indexes*[^indexes-footnote], some
subset of the elements of `a`. An index `a[x]` always either returns a new
array containing a subset of the elements of `a` or raises an `IndexError`.
When it comes to indexing, the most important rule, which applies to all types
of indices, is this:

[^indexes-footnote]: For clarity, in this document and throughout the ndindex
    documentation, the plural of *index* is *indices*. *Indexes* is always a
    verb. For example,

    > In `a[i, j]`, the *indices* are `i` and `j`. They represent a single
      tuple index `(i, j)`, which *indexes* the array `a`.

> **Indices do not in any way depend on the *values* of the elements they
  select. They only depend on their *positions* in the array `a`.**

For example, consider `a`, an array of integers with the shape `(2, 3, 2)`:

```py
>>> import numpy as np
>>> a = np.array([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
>>> a.shape
(2, 3, 2)
```

Let's take as an example the index `0, ..., 1:`. We'll investigate how
exactly this index works later. For now, just notice that `a[0, ..., 1:]`
returns a new array with some of the elements of `a`.

```py
>>> a[0, ..., 1:]
array([[1],
       [3],
       [5]])
```

Now consider another array, `b`, with the exact same shape `(2, 3, 2)`, but
containing completely different entries, such as strings. If we apply the same
index `0, ..., 1:` to `b`, it will choose the exact same corresponding
elements.

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

Therefore, the following statements are always true about any index:

- **An index on an array always produces a new array with the same dtype (unless
  it raises `IndexError`).**

- **Each element of the new array corresponds to some element of the original
  array.**

- **These elements are chosen by their position in the original array only.
  The values of these elements are irrelevant.**

- **As such, the same index applied to any other array with the same shape will
  produce an array with the exact same resulting shape with elements in the
  exact same corresponding places.**

The full range of valid indices allows the generation of more or less
arbitrary new arrays whose elements come from the indexed array `a`. In
practice, the most commonly desired indexing operations are represented by
basic indices such as [integer indices](integer-indices.md),
[slices](slices.md), and [ellipses](multidimensional-indices/ellipses.md).

## Footnotes
