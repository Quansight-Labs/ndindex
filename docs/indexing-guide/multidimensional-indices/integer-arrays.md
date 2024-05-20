# Integer Array Indices

```{note}
In this section, and [the next](boolean-arrays), do not confuse the *array
being indexed* with the *array that is the index*. The former can be anything
and have any dtype. It is only the latter that is restricted to being integer
or boolean.
```

Integer array indices are very powerful. Using them, you can effectively
construct arbitrary new arrays consisting of elements from the original
indexed array.


To start, let's consider a simple one-dimensional array:

```py
>>> import numpy as np
>>> a = np.array([100, 101, 102, 103])
```

Now suppose we wish to construct the following 2-D array from this array using
only indexing operations:

```
[[ 100, 102, 100 ],
 [ 103, 100, 102 ]]
```

It should hopefully be clear that there's no way we could possibly construct
this array as `a[idx]` using only the index types we've discussed so far. For
one thing, [integer indices](../integer-indices.md), [slices](../slices.md),
[ellipses](ellipses.md), and [newaxes](newaxis.md) all only select
elements of the array in order (or possibly [reversed order](negative-steps)
for slices), whereas this array has elements completely shuffled from `a`, and
some are even repeated.

However, we could "cheat" a bit here, and do something like

```py
>>> new_array = np.array([[a[0], a[2], a[0]],
...                       [a[3], a[0], a[2]]])
>>> new_array
array([[100, 102, 100],
       [103, 100, 102]])
```

This is the array we want. We sort of constructed it using only indexing
operations, but we didn't actually do `a[idx]` for some index `idx`. Instead,
we just listed the index of each individual element.

An integer array index is essentially this "cheating" method, but as a single
index. Instead of listing out `a[0]`, `a[2]`, and so on, we just create a
single integer array with those [integer indices](../integer-indices.md):

```py
>>> idx = np.array([[0, 2, 0],
...                 [3, 0, 2]])
```

If we then index `a` with this array, it works just like `new_array` above:

```py
>>> a[idx]
array([[100, 102, 100],
       [103, 100, 102]])
```

(multidimensional-integer-indices)=
This is how integer array indices work:

> **An integer array index can construct *arbitrary* new arrays with elements
from `a`, with the elements in any order and even repeated, simply by
enumerating the integer index positions where each element of the new array
comes from.**

Note that `a[idx]` above is not the same size as `a` at all. `a` has 4
elements and is 1-dimensional, whereas `a[idx]` has 6 elements and is
2-dimensional. `a[idx]` also contains some duplicate elements from `a`, and
there are some elements which aren't selected at all. Indeed, we could take
*any* integer array of any shape, and as long as the elements are between 0
and 3, `a[idx]` would create a new array with the same shape as `idx` with
corresponding elements selected from `a`.

A useful way to think about integer array indexing is that it generalizes
[integer indexing](../integer-indices.md). With integer indexing, we are
effectively indexing using a 0-dimensional integer array, that is, a single
integer.[^integer-scalar-footnote] This always selects the corresponding
element from the given axis and removes the dimension. That is, it replaces
that dimension in the shape with `()`, the "shape" of the integer index.

Similarly,

> **an integer array index `a[idx]` selects elements from the specified axis
> and replaces the dimension in the shape with the shape of the index array
> `idx`.**


For example:

```
>>> a = np.empty((3, 4))
>>> idx = np.zeros((2, 2), dtype=int)
>>> a[idx].shape # (3,) is replaced with (2, 2)
(2, 2, 4)
>>> a[:, idx].shape # Indexing the second dimension, (4,) is replaced with (2, 2)
(3, 2, 2)
```

In particular, even when the index array `idx` has more than one dimension, an
integer array index still only selects elements from a single axis of `a`. It
would appear that this limits the ability to arbitrarily shuffle elements of
`a` using integer indexing. For instance, suppose we want to create the array
`[105, 100]` from the above 2-D `a`. Based on the above examples, it might not
seem possible, since the elements `105` and `100` are not in the same row or
column of `a`.

However, this is doable by providing multiple integer array
indices:

(multiple-integer-arrays)=
> **When multiple integer array indices are provided, the elements of each
> index are selected correspondingly for that axis.**

It's perhaps most illustrative to show this as an example. Given the above
`a`, we can produce the array `[105, 100]` using

```
>>> a = np.array([[100, 101, 102],
...               [103, 104, 105]])
>>> idx = (np.array([1, 0]), np.array([2, 0]))
>>> a[idx]
array([105, 100])
```

Let's break this down. `idx` is a [tuple index](tuples.md) with two arrays,
which are both the same shape. The first element of our desired result, `105`
corresponds to index `(1, 2)` in `a`:

```py
>>> a[1, 2] # doctest: +SKIPNP1
np.int64(105)
```

So we write `1` in the first array and `2` in the second array. Similarly, the
next element, `100` corresponds to index `(0, 0)`, so we write `0` in the
first array and `0` in the second. In general, the first array contains the
indices for the first axis, the second array contains the indices for the
second axis, and so on. If we were to
[zip](https://docs.python.org/3/library/functions.html#zip) up our two index
arrays, we would get the set of indices for each corresponding element, `(1,
2)` and `(0, 0)`.

The resulting array has the same shape as our two index arrays. As before,
this shape can be arbitrary. Suppose we want to create the array

```
[[[ 102, 103],
  [ 102, 101]],
 [[ 100, 105],
  [ 102, 102]]]
```

Recall our array `a`:

```
>>> a
array([[100, 101, 102],
       [103, 104, 105]])
```

Noting the index for each element in our desired array, we get

```
>>> idx0 = np.array([[[0, 1], [0, 0]], [[0, 1], [0, 0]]])
>>> idx1 = np.array([[[2, 0], [2, 1]], [[0, 2], [2, 2]]])
>>> a[idx0, idx1]
array([[[102, 103],
        [102, 101]],
<BLANKLINE>
       [[100, 105],
        [102, 102]]])
```

Again, reading across, the first element, `102` corresponds to index `(0, 2)`,
the next element, `103`, corresponds to index `(1, 0)`, and so on.

## Use Cases

A common use case for integer array indexing is sampling. For example, to
sample $k$ elements from a 1-D array of size $n$ with replacement, we can
simply construct an a random integer index in the range $[0, n)$ with $k$
elements (see the
{external+numpy:meth}`numpy.random.Generator.integers`
documentation):[^random-integers-footnote]

[^random-integers-footnote]: Note that `np.random` also supports this
    operation directly with
    {external+numpy:meth}`numpy.random.Generator.choice`.

```
>>> k = 10
>>> a = np.array([100, 101, 102, 103]) # as above
>>> rng = np.random.default_rng(11) # Seeded so this example reproduces
>>> idx = rng.integers(0, a.size, k) # rng.integers() excludes the upper bound
>>> idx
array([0, 0, 3, 1, 2, 2, 2, 0, 1, 0])
>>> a[idx]
array([100, 100, 103, 101, 102, 102, 102, 100, 101, 100])
```

(permutation-example)=

Another common use case of integer array indexing is to permute an array. An
array can be randomly permuted with
{external+numpy:meth}`numpy.random.Generator.permutation`. But what if we want
to permute two arrays with the same permutation? We can compute a permutation
index and apply it to both arrays. For a 1-D array `a` of size $n$, a
permutation index is just a permutation of the integer array index
`np.arange(n)`, which itself is the [identity
permutation](https://en.wikipedia.org/wiki/identity_permutation) on `a`:

```py
>>> a = np.array([100, 101, 102, 103]) # as above
>>> b = np.array([200, 201, 202, 203]) # another array
>>> identity = np.arange(a.size)
>>> a[identity] # arange by itself is the identity permutation index
array([100, 101, 102, 103])
>>> rng = np.random.default_rng(11) # Seeded so this example reproduces
>>> random_permutation = rng.permutation(identity)
>>> a[random_permutation]
array([103, 101, 100, 102])
>>> b[random_permutation] # The same permutation on b
array([203, 201, 200, 202])
```

(integer-arrays-advanced-notes)=
## Advanced Notes

The information above provides the basic gist of integer array indexing, but
there are also many subtleties and advanced behaviors involved with them. The
subsections here are included for completeness; however, if you are a beginner
NumPy user, you may wish to skip them.

### Negative Indices

> **Indices in the integer array can also be negative. Negative indices work
the same as they do with [integer indices](../integer-indices.md).**

Negative and
nonnegative indices can be mixed arbitrarily.

```py
>>> a = np.array([100, 101, 102, 103]) # as above
>>> idx = np.array([0, 1, -1])
>>> a[idx]
array([100, 101, 103])
```

If you want to convert an index containing negative indices into an index
without any negative indices, you can use the ndindex
[`reduce()`](ndindex.IntegerArray.reduce) method with a `shape` argument.

### Python Lists

> **You can use a list of integers instead of an array to represent an integer
array index.[^lists-footnote]**

Using a list is useful when writing an array index by hand; however, in all
other cases, using an actual array is preferable. In most real-world
scenarios, an array index is constructed from some other array methods.

[^lists-footnote]: Beware that [versions of NumPy prior to
    1.23](https://numpy.org/doc/stable/release/1.23.0-notes.html#expired-deprecations)
    treated a single list as a [tuple index](tuples.md) rather than as an
    array.


```py
>>> a = np.array([100, 101, 102, 103]) # as above
>>> a[[0, 1, -1]]
array([100, 101, 103])
>>> idx = np.array([0, 1, -1])
>>> a[idx] # this is the same
array([100, 101, 103])
```

(integer-array-broadcasting)=
### Broadcasting

> **The integer arrays in an index must either be the same shape or be able to
> be [broadcast](broadcasting) together to the same shape.**

If the arrays are not the same shape, they are first broadcast together, and
those broadcasted arrays are used as the indices. This broadcasting behavior
is useful if the index array would otherwise be repeated in a given dimension,
and provides a convenient way to do outer indexing (see the [next
section](outer-indexing)).

This also means that mixing an integer array index with a single [integer
index](../integer-indices.md) is the same as replacing the single integer
index with an array of the same shape filled with that integer (because
remember, a single integer index is the same thing as an integer array index
of shape `()`).

For example:

```py
>>> a = np.array([[100, 101, 102],  # as above
...               [103, 104, 105]])
>>> idx0 = np.array([1, 0])
>>> idx0.shape
(2,)
>>> idx1 = np.array([[0], [1], [2]])
>>> idx1.shape
(3, 1)
>>> # idx0 and idx1 broadcast to shape (3, 2), which will
>>> # be the shape of a[idx0, idx1]
>>> a[idx0, idx1]
array([[103, 100],
       [104, 101],
       [105, 102]])
>>> a[idx0, idx1].shape
(3, 2)
>>> idx0_broadcasted = np.array([[1, 0], [1, 0], [1, 0]])
>>> idx1_broadcasted = np.array([[0, 0], [1, 1], [2, 2]])
>>> idx0_broadcasted.shape
(3, 2)
>>> idx1_broadcasted.shape
(3, 2)
>>> a[idx0_broadcasted, idx1_broadcasted] # The same thing as a[idx0, idx1]
array([[103, 100],
       [104, 101],
       [105, 102]])
```

(mixing-array-and-integer)=
And mixing an array and an integer index:

```py
>>> a
array([[100, 101, 102],
       [103, 104, 105]])
>>> idx0 = np.array([1, 0, 0])
>>> a[idx0, 2]
array([105, 102, 102])
>>> idx1_broadcasted = np.array([2, 2, 2]) # The 0-D array '2' broadcasted to shape (3,)
>>> a[idx0, idx1_broadcasted] # The same thing as a[idx0, 2]
array([105, 102, 102])
```

Here the `idx0` array specifies the indices along the first dimension, `1`,
`0`, and `0`, and the `2` specifies to always use index `2` along the second
dimension. This is the same as using the array `[2, 2, 2]` for the second
dimension, since this is the scalar `2` broadcasted to the shape of `[1, 0,
0]`.

The ndindex methods
[`Tuple.broadcast_arrays()`](ndindex.Tuple.broadcast_arrays) and
[`expand()`](ndindex.Tuple.expand) will broadcast array indices together into
a canonical form.

[^integer-scalar-footnote]:
    <!-- This is the only way to cross reference a footnote across documents -->
    (integer-scalar-footnote-ref)=

    In fact, if the integer array index itself has
    shape `()`, then the behavior is identical to simply using an `int` with
    the same value. So it's a true generalization. In ndindex,
    [`IntegerArray.reduce()`](ndindex.IntegerArray.reduce) will always convert
    a 0-D array index into an [`Integer`](ndindex.integer.Integer).

    However, there is one difference between `a[0]` and `a[asarray(0)]`. The
    latter is considered an advanced index, so it does not create a
    [view](views-vs-copies):

    ```py
    >>> a = np.empty((2, 3))
    >>> a[0].base is a
    True
    >>> print(a[np.array(0)].base)
    None
    ```

(outer-indexing)=
#### Outer Indexing

The broadcasting behavior for multiple integer indices may seem odd, but it
serves a useful purpose. [As we saw above](multiple-integer-arrays), multiple
integer array indices are required to select elements from higher dimensional
arrays, one array for each dimension. These integer arrays enumerate the
indices of the selected elements along these dimensions. For example, as
above:

```py
>>> a = np.array([[100, 101, 102],
...               [103, 104, 105]])
>>> a[[1, 0], [2, 0]] # selects elements (1, 2) and (0, 0)
array([105, 100])
```

However, you might have noticed that this behavior is somewhat unusual
compared to other index types. For all other index types we've discussed so
far, such as [slices](../slices.md) and [integer indices](../integer-indices.md),
each index applies "independently" along each dimension. For example, `x[0:2,
0:3]` applies the slice `0:2` to the first dimension of `x` and `0:3` to the
second dimension. The resulting array has `2*3 = 6` elements, because there
are 3 subarrays selected from the first dimension with 2 elements each. But in
the above example, `a[[1, 0], [2, 0]]` only has 2 elements, not 4. And
something like `a[[1, 0], [2, 0, 1]]` is an error.

The integer array equivalent of the way slices work is called "outer
indexing".[^vectorized-indexing-footnote] An outer index "`a[[1, 0], [2, 0, 1]]`" would have 6 elements: rows
1 and 0, with elements from columns 2, 0, and 1, in that order. However, the
index `a[[1, 0], [2, 0, 1]]` doesn't actually work like
this.[^outer-indexing-footnote]

[^vectorized-indexing-footnote]: The type of integer array indexing that NumPy
    uses where arrays are broadcasted and "zipped" together is sometimes called
    "vectorized indexing" or "inner indexing". The "outer" and "inner" are
    because they act like an outer- or inner-product.

[^outer-indexing-footnote]: Outer indexing is how integer array indexing works
    in many other languages such as MATLAB, Fortran, and R. There is a proposed
    [NEP](https://numpy.org/neps/nep-0021-advanced-indexing.html) to add more
    direct support for outer indexing like this to NumPy, but it hasn't been
    accepted yet.

Strictly speaking, though, NumPy's integer array indexing rules do allow for
outer indexing. This is because, as we saw above, integer array indexing
allows for creating *arbitrary* new arrays from a given input array. And as it
turns out, the integer arrays required to represent an outer array index are
quite simple to construct. They are simply the outer index arrays broadcasted
together.

To see why this is, consider the above example, `a[[1, 0], [2, 0, 1]]`. We
want our end result to be

```
[[105, 103, 104],
 [102, 101, 100]]
```

That is, the rows of `a` should be in the order `[0, 1]`, and the columns
should be in the order `[2, 0, 1]`. The end result should be an array of shape
`(2, 3)` (which happens to be the same shape as `a`, but that's just a
coincidence; an outer-indexed array constructed from `a` could have any 2-D
shape). So using the integer array indexing rules above, we need to index `a`
by integer arrays of shape `(2, 3)`. Since `a` has two dimensions, we will
need two index arrays, one for each dimension. Let's consider what these
arrays should be. For the first dimension, we want to select row `1` three
times and then row `0` three times:

```
[[1, 1, 1],
 [0, 0, 0]]
```

And for the second dimension, we want to select the columns `2`, `0`, and `1`,
in that order, regardless of which row we are in:

```
[[2, 0, 1],
 [2, 0, 1]]
```

In general, we want to repeat each outer selection array along the
corresponding dimension so as to fill an array with the final desired shape.
This is exactly what broadcasting does! If we reshape our first array to have
shape `(2, 1)` and the second array to have shape `(1, 3)`, then broadcasting
them together will repeat the first dimension of the first array along the
second axis, and the second dimension of the second array along the first
axis, i.e., exactly the arrays we want.

This is why NumPy automatically broadcasts integer array indices together.

> **Outer indexing arrays can be constructed by inserting size-1 dimensions
> into the desired "outer" integer array indices so that the non-size-1
> dimension for each is in the indexing dimension.**

For example,

```py
>>> idx0 = np.array([1, 0])
>>> idx1 = np.array([2, 0, 1])
>>> a[idx0[:, np.newaxis], idx1[np.newaxis, :]]
array([[105, 103, 104],
       [102, 100, 101]])
```

Here, we use [newaxis](newaxis.md) along with `:` to turn `idx0` and
`idx1` into shape `(2, 1)` and `(1, 3)` arrays, respectively. These then
automatically broadcast together to give the desired outer index.

This "insert size-1 dimensions" operation can also be performed automatically
with the {external+numpy:func}`numpy.ix_` function.[^ix-footnote]

[^ix-footnote]: `ix_()` is currently limited to only support 1-D input arrays
    and can't be mixed with other index types. In the general case you will
    need to apply the reshaping operation manually. There is an [open
    issue](https://github.com/Quansight-Labs/ndindex/issues/29) to implement
    this more generally in ndindex.

```py
>>> np.ix_(idx0, idx1)
(array([[1],
       [0]]), array([[2, 0, 1]]))
>>> a[np.ix_(idx0, idx1)]
array([[105, 103, 104],
       [102, 100, 101]])
```

Outer indexing can be thought of as a generalization of slicing. With a
[slice](../slices.md), you can really only select a "regular" sequence of
elements from a dimension, that is, either a contiguous chunk, or a contiguous
chunk split by a regular step value. It's impossible, for instance, to use a
slice to select the indices `[0, 1, 2, 3, 5, 6, 7]`, because `4` is omitted.
For instance, say the first dimension of your array represents time steps and
you want to select time steps 0--7, but time step 4 is invalid for some reason
and you want to ignore it for your analysis. If you just care about the first
dimension, you can just use the integer index `[0, 1, 2, 3, 5, 6, 7]`. But
suppose you also wanted select some other non-contiguous "slice" from the
second dimension. Using just basic indices, you'd have to index the array with
normal slices then either remove or ignore the non-desired indices, neither of
which is ideal. And it would be even more complicated if you also wanted the
indices out-of-order or repeated for some reason.

With outer indexing, you would just construct your "slice" of non-contiguous
indices as integer arrays, turn them into "outer" indices using `ix_` or
manual reshaping, then use that outer index to construct the desired array
directly.

Conversely, a slice like `2:9` is equivalent to the outer index `[2, 3,
4, 5, 6, 7, 8]`.[^slice-outer-index-footnote]

[^slice-outer-index-footnote]: They aren't actually equivalent, because [a
    slice creates a view and an integer array index creates a
    copy](views-vs-copies), not to mention the fact that slices
    [clip](clipping) and integer arrays have bounds checks. If your index can
    be represented as a slice, it's usually better to use an actual `slice`.

### Assigning to an Integer Array Index

As with all index types discussed in this guide, an integer array index can be
used on the left-hand side of an assignment. This is useful because it allows
you to surgically inject new elements into existing positions in your array.

```py
>>> a = np.array([100, 101, 102, 103]) # as above
>>> idx = np.array([0, 3])
>>> a[idx] = np.array([200, 203])
>>> a
array([200, 101, 102, 203])
```

However, exercise caution, as this is inherently ambiguous if the index array
contains duplicate elements. For example, suppose we attempted to
set index `0` to both `1` and `3`:

```py
>>> a = np.array([100, 101, 102, 103]) # as above
>>> idx = np.array([0, 1, 0])
>>> a[idx] = np.array([1, 2, 3])
>>> a
array([  3,   2, 102, 103])
```

The end result was `3`. This happened because `3` corresponded to the last `0`
in the index array. But importantly, this is just an implementation detail.
**NumPy makes no guarantees regarding the order in which index elements are
assigned.**[^cupy-assignment-footnote] If you are using an integer array as an
assignment index, be careful to avoid duplicate entries in the index or, at
the very least, ensure that duplicate entries are always assigned the same
value.

[^cupy-assignment-footnote]: For example, in [CuPy](https://cupy.dev/),
  which implements the NumPy API on top of GPUs, [the behavior of this sort
  of thing is
  undefined](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__setitem__).
  This is because CuPy parallelizes the assignment operation on the GPU, and
  the element that gets assigned to the duplicate index "last" becomes
  dependent on a race condition.

(integer-arrays-combined-with-basic-indices)=
### Combining Integer Arrays Indices with Basic Indices

If any [slice](../slices.md), [ellipsis](ellipses.md), or
[newaxis](newaxis.md) indices precede or follow all the
[integer](../integer-indices) and integer array indices in an index, the two
sets of indices operate independently. Slices and ellipses select the
corresponding axes, newaxes add new axes to these locations, and the integer
array indices select the elements on their respective axes, as previously
described.

For example, consider:

```py
>>> a = np.array([[[100, 101, 102],  # Like above, but with an extra dimension
...                [103, 104, 105]]])
```

This is the same `a` as in the above examples, except it has an extra size-1
dimension:

```py
>>> a.shape
(1, 2, 3)
```

We can select this first dimension with a slice `:`, then use the exact same
index as in the example [shown previously](mixing-array-and-integer):

```py
>>> idx0
array([1, 0])
>>> a[:, idx0, 2]
array([[105, 102]])
>>> a[:, idx0, 2].shape
(1, 2)
```

The primary point of this behavior is that you can use `...` at the beginning
of an index to select the last axes of an array using integer array indices,
or several `:`s to select some middle axes. This lets you do with indexing
what you can also do with the {external+numpy:func}`numpy.take` function.

To be sure though, this index could use any slice, not just `:`, and could
also include newaxes. This behavior is mainly implemented for the sake of
semantic completeness, although it could potentially allow combining two
sequential indexing operations into a single step.

### Integer Array Indices Separated by Basic Indices

Finally, if the [slices](../slices.md), [ellipses](ellipses.md), or
[newaxes](newaxis.md) are *in between* the integer array indices, then
something more strange happens. The two index types still operate
"independently"; however, unlike the previous case, the shape derived from the
array indices is *prepended* to the shape derived from the non-array indices.
This is because in these cases there is inherent ambiguity in where these
dimensions should be placed in the final shape.

An example demonstrates this most clearly:

```py
>>> a = np.empty((2, 3, 4, 5))
>>> a.shape
(2, 3, 4, 5)
>>> idx = np.zeros((10, 20), dtype=int)
>>> idx.shape
(10, 20)
>>> a[idx, :, :, idx].shape
(10, 20, 3, 4)
```

Here the integer array index shape `(10, 20)` comes first in the result array
and the shape corresponding to the rest of the index, `(3, 4)`, comes last.

If you find yourself running into this behavior, chances are you would be
better off rewriting the indexing operation to be simpler, for instance, by
first reshaping the array so that the integer array indices are together in
the index. This is considered a design flaw in
NumPy[^advanced-indexing-design-flaw-footnote], and no other Python array
library has replicated it. ndindex will raise a `NotImplementedError`
exception on indices like these, because I don't want to deal with
implementing this obscure
logic.[^ndindex-advanced-indexing-design-flaw-footnote]

[^advanced-indexing-design-flaw-footnote]: Travis Oliphant, the original
    creator of NumPy, told me privately that "somebody should have slapped me
    with a wet fish" when he designed this.

[^ndindex-advanced-indexing-design-flaw-footnote]: I might accept a pull
    request implementing it, but I'm not going to do it myself.

## Exercise

Based on the above sections, you should be able to complete the following
exercise: How might you randomly permute a 2-D array using
{external+numpy:meth}`numpy.random.Generator.permutation` and indexing, so
that each axis is permuted independently? This operation might correspond to
multiplying the array by random [permutation
matrices](https://en.wikipedia.org/wiki/Permutation_matrix) on the left and
right, like $P_1AP_2$.

For example, the array


```py
a = array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
```

Might be permuted to

```py
a_perm = array([[ 5,  4,  6,  7],
                [ 1,  0,  2,  3],
                [ 9,  8, 10, 11]])
```

(Note that this is not a full permutation of the array. For instance, the
first row `[5, 4, 7, 6]` contains only elements from the second row of `a`.)

~~~~{dropdown} Click here to show the solution

Suppose we start with the following 2-D array `a`:

```py
>>> a = np.arange(12).reshape((3, 4))
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
```

We can generate permutations for the two axes using
{external+numpy:meth}`numpy.random.Generator.permutation` [as
above](permutation-example):

```py
>>> rng = np.random.default_rng(11) # Seeded so this example reproduces
>>> idx0 = rng.permutation(np.arange(3))
>>> idx1 = rng.permutation(np.arange(4))
```

However, we cannot do `a[idx0, idx1]` as this will fail.

```py
>>> a[idx0, idx1]
Traceback (most recent call last):
...
IndexError: shape mismatch: indexing arrays could not be broadcast together
with shapes (3,) (4,)
```

Remember that we want a permutation of `a`, so the result array should have
the same shape as `a` (`(3, 4)`). This should therefore be the broadcasted
shape of `idx0` and `idx1`, which are currently shapes `(3,)`, and `(4,)`. We
can use [`newaxis`](newaxis.md) to insert dimensions so that they are
shape `(3, 1)` and `(1, 4)` so that they broadcast together to this shape.

```py
>>> a[idx0[:, np.newaxis], idx1[np.newaxis]]
array([[ 5,  4,  6,  7],
       [ 1,  0,  2,  3],
       [ 9,  8, 10, 11]])
```

You can check that this is a permutation of `a` where each axis is permuted
independently.

We can also interpret this as an [outer indexing](outer-indexing) operation.
In this case, our non-contiguous "slices" that we are outer indexing by are a
full slice along each axis, just permuted. We can use the `ix_()` helper to
construct the same index as above

```
>>> a[np.ix_(idx0, idx1)]
array([[ 5,  4,  6,  7],
       [ 1,  0,  2,  3],
       [ 9,  8, 10, 11]])
```

As an extra bonus, here's how we can interpret this as a multiplication by
permutation matrices, using the same indices (but of course, simply permuting
`a` directly with the indices is more efficient):

```py
>>> P1 = np.eye(3, dtype=int)[idx0]
>>> P2 = np.eye(4, dtype=int)[idx1]
>>> P1 @ a @ P2.T
array([[ 5,  4,  6,  7],
       [ 1,  0,  2,  3],
       [ 9,  8, 10, 11]])
```

Can you see why this works?
~~~~

```{rubric} Footnotes
```
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
