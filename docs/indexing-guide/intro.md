(what-is-an-index)=
# Introduction: What is an Index?

Nominally, an index is any object that can go between the square brackets
after an array. That is, if `a` is a NumPy array, then in `a[x]`, *`x`* is an
*index* of `a`.[^index-vs-slice-footnote] This also applies to built-in
sequence types in Python such as `list`, `tuple`, and `str`, but be careful to
not confuse the same notation used on Python dictionaries. If `d` is a Python
dictionary, it uses the same notation `d[x]`, but the meaning of `x` is
completely different than what is being discussed in this document (and
indeed, many index types will not even work if you try them on a dictionary).
This document also does not apply to indexing Pandas DataFrame or Series
objects, except insomuch as they reuse the same semantics as NumPy. Finally,
note that some other Python array libraries (e.g., PyTorch or Jax) have
similar indexing rules, but most only implement a subset of the full NumPy
semantics outlined here.

[^index-vs-slice-footnote]: Some people call `x` a *slice* of `a`, but we
    avoid this confusing nomenclature, using *slice* to refer only to the
    [slice index type](slices-docs). The term "index" is used in the Python
    language itself (e.g., in the built-in exception type `IndexError`).

Semantically, an index `x` selects, or *indexes*[^indexes-footnote], some
subset of the elements of `a`. An index `a[x]` always either returns a new
array with some subset of the elements of `a`, or it raises `IndexError`. The
most important rule for indexing, which applies to all types of indices, is
this:

[^indexes-footnote]: For clarity, in this document, and throughout the ndindex
    documentation, the plural of *index* is *indices*. *Indexes* is always a
    verb. For example,

    > In `a[i, j]`, the *indices* are `i` and `j`. They represent a single
      tuple index `(i, j)`, which *indexes* the array `a`.

> **Indices do not in any way depend on the *values* of the elements they
  select. They only depend on their *positions* in the array `a`.**

For example, suppose `a` is an array of integers of shape `(2, 3, 2)`:

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

Now take another array, `b`, with the exact same shape `(2, 3, 2)`, but
completely different entries, say, strings. If we apply the same index `0,
..., 1:` to `b`, it will choose the exact same corresponding elements.

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

- **An index on an array always produces a new array with the same dtype (unless
  it raises `IndexError`).**

- **Each element of the new array corresponds to some element of the original
  array.**

- **These elements are chosen by their position in the original array only.
  Their values are irrelevant.**

- **As such, the same index applied to any other array with the same shape will
  produce an array with the exact same resulting shape with elements in the
  exact same corresponding places.**

The full range of valid indices allow generating more or less arbitrary new
arrays whose elements come from the indexed array `a`. In practice, the most
commonly desired indexing operations are represented by the basic indices such
as [integer indices](integer-indices), [slices](slices-docs), and
[ellipses](ellipsis-indices).
