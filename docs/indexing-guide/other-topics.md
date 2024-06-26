# Other Topics Relevant to Indexing

There is a great deal of functionality in NumPy, and most of it is
out-of-scope for this guide. However, a few additional features are useful to
understand in order to fully utilize indexing to its fullest potential, or as
important caveats to be aware of when doing indexing operations.

(broadcasting)=
## Broadcasting

Broadcasting is a powerful abstraction that applies to all operations in
NumPy. It allows arrays with mismatched shapes to be combined together as if
one or more of their dimensions were simply repeated the appropriate number of
times.

Normally, when we perform an operation on two arrays with the same shape, it
does what we'd expect, i.e., the operation is applied to each corresponding
element of each array. For example, if `x` and `y` are both `(2, 2)` arrays,
`x + y` results in a `(2, 2)` array with the sum of the corresponding
elements.

```py
>>> import numpy as np
>>> a = np.array([[1, 2],
...               [3, 4]])
>>> b = np.array([[101, 102],
...               [103, 104]])
>>> a + b
array([[102, 104],
       [106, 108]])
```

However, you may have noticed that `x` and `y` don't always have to be two
arrays with the same shape. For example, you can add a single scalar element
to an array, and it will add it to each element.

```py
>>> a + 1
array([[2, 3],
       [4, 5]])
```

In the above example, we can think of the scalar `1` as a shape `()` array,
whereas `x` is a shape `(2, 2)` array. Thus, `x` and `1` do not have the same
shape, but `x + 1` is allowed via repeating `1` across every element of `x`.
This means NumPy is taking `1` and treating it as if it were the shape `(2,
2)` array `[[1, 1], [1, 1]]`.

Broadcasting is a generalization of this behavior. Specifically, instead of
repeating just a single number into an array, we can repeat just some
dimensions of an array into a bigger array. For example, here we multiply `x`,
a shape `(3, 2)` array, with `y`, a shape `(2,)` array. `y` is virtually
repeated into a shape `(3, 2)` array with each element of the last dimension
repeated 3 times.

```py
>>> x = np.array([[1, 2],
...               [3, 4],
...               [5, 6]])
>>> x.shape
(3, 2)
>>> y = np.array([0, 2])
>>> x*y
array([[ 0,  4],
       [ 0,  8],
       [ 0, 12]])
```

We can see how broadcasting works using `np.broadcast_to`

```py
>>> np.broadcast_to(y, x.shape)
array([[0, 2],
       [0, 2],
       [0, 2]])
```

This is what the array `y` looks like before it is combined with `x`. Although
note that broadcasting is implemented efficiently, so that the "repeated"
elements actually refer to the same objects in memory. See [](views-vs-copies)
and [](strides) below.

Broadcasting always happens automatically in NumPy whenever two arrays with
different shapes are combined using any function or operator, assuming those
shapes are broadcast compatible. The rule for broadcast compatibility is that
the shorter of the shapes are prepended with length 1 dimensions so that they
have the same number of dimensions. Then any dimensions that are size `1` in a
shape are replaced with the corresponding size in the other shape. The other
non-`1` sizes must be equal or broadcasting is not allowed.

In the above example, we broadcast `(3, 2)` with `(2,)` by first extending
`(2,)` to `(1, 2)` then broadcasting the size `1` dimension to the
corresponding size in the other shape, `3`, giving a broadcasted shape of `(3,
2)`. In more advanced examples, both shapes may have broadcasted dimensions.
For instance, `(3, 1)` can broadcast with `(2,)` to create `(3, 2)`. The first
shape repeats the first axis 2 times along the second axis, while the second
shape inserts a new axis at the beginning that repeats 3 times.

We can think of `(3, 2)` as a "stack" of 3 shape `(2,)` arrays. Just as the
scalar `1` got repeated to match the full shape of `a` above, the shape `(2,)`
array `y` gets repeated into a `(3,)` "stack" so it matches the shape of `x`.

Additionally, size `1` dimensions are a signal to NumPy that that dimension is
allowed to be repeated, or broadcasted, when used with another array where
the other dimensions match.

It can be useful to think of broadcasting as repeating "stacks" of smaller
arrays in this way. The automatic prepending rule lets you automatically treat
array "stacking" along the last dimension, which is the most common, but the
size `1` dimension rule allows these stacks to be along any dimensions of
the array, not just the last ones.

When it comes to indexing, one of the most useful types of index for use with
broadcasting is [newaxis](./multidimensional-indices/newaxis.md), which lets
you easily insert size `1` dimensions into an array to make the broadcastable
in a specific way. See [](where-newaxis-is-used).

See the [NumPy
documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html) for
more examples of broadcasting.

(views-vs-copies)=
## Views vs. Copies

There is an important distinction between basic indices (i.e.,
[integers](integer-indices.md), [slices](slices.md),
[ellipses](multidimensional-indices/ellipses.md),
[newaxis](multidimensional-indices/newaxis.md)) and [advanced
indices](advanced-indices) (i.e., [integer array
indices](multidimensional-indices/integer-arrays.md) and [boolean array
indices](multidimensional-indices/boolean-arrays.md)) in NumPy that must be
noted in some situations. Specifically, basic indices always create a **view**
into an array[^view-scalar-footnote], while advanced indices always create a
**copy** of the underlying array. [Tuple](multidimensional-indices/tuples.md)
indices (i.e., multidimensional indices) will create a view if they do not
contain an advanced index and a copy if they do.


[^view-scalar-footnote]:
    <!-- This is the only way to cross reference a footnote across documents -->
    (view-scalar-footnote-ref)=

    There is one exception to this rule, which is that an
    index that would return a scalar returns a copy, since scalars are
    supposed to be immutable.

    ```py
    >>> a = np.arange(20)
    >>> a[0] # doctest: +SKIPNP1
    np.int64(0)
    >>> print(a[0].base)
    None
    >>> a[0, ...]
    array(0)
    >>> a[0, ...].base is a
    True
    ```

A **view** is a special type of array whose data points to another array. This
means that if you mutate the data in one array, the other array will also have
that data mutated as well. For example,

```py
>>> a = np.arange(24).reshape((3, 2, 4))
>>> b = a[:, 0] # b is a view of a
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> b[:] = 0 # Mutating b also mutates a
>>> a
array([[[ 0,  0,  0,  0],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 0,  0,  0,  0],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[ 0,  0,  0,  0],
        [20, 21, 22, 23]]])
```

Mutating `b` also changed `a`, because both arrays point to the same memory.

Note that this behavior is exactly the opposite of the way Python lists work.
With Python lists, `a[:]` is a shorthand to copy `a`. But with NumPy, `a[:]`
creates a view into `a` (to copy an array with NumPy, use `a.copy()`). Python
lists do not have a notion of views.

```py
>>> a = [1, 2, 3] # list
>>> b = a[:] # a copy of a
>>> b[0] = 0 # Modifies b but not a
>>> b
[0, 2, 3]
>>> a
[1, 2, 3]
>>> a = np.array([1, 2, 3]) # NumPy array
>>> b = a[:] # A view of a
>>> b[0] = 0 # Modifies both b and a
>>> b
array([0, 2, 3])
>>> a
array([0, 2, 3])
>>> c = a.copy() # A copy of a
>>> c[0] = -1 # Only modifies c
>>> c
array([-1, 2, 3])
>>> a
array([0, 2, 3])
```

Views don't just come from indexing. For instance, when you reshape an array,
that also creates a view.

```py
>>> a = np.arange(24)
>>> b = a.reshape((3, 2, 4))
>>> b[0] = 0
>>> b
array([[[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> a
array([ 0,  0,  0,  0,  0,  0,  0,  0,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23])
```

Many other operations also create views, for example
[`np.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html),
[`a.T`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html),
[`np.ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html),
[`broadcast`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html),
and
[`a.view`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html).[^view-functions-footnote]

[^view-functions-footnote]: Some of these functions will sometimes return a
    copy because returning a view is not possible. For example, it is not
    always possible to represent a reshape as a [strides](strides)
    manipulation if the strides are already non-contiguous.

To check if an array is a view, check `a.base`. It will be `None` if it is not
a view and will point to the base array otherwise. A view of a view will have
the same base as the original array.

```py
>>> a = np.arange(24)
>>> b = a[:]
>>> print(a.base) # a is not a view
None
>>> b.base is a # b is a view of a
True
>>> c = b[::2]
>>> c.base is a # c is a further view of a
True
```

In contrast, an advanced index will always create a copy (even if it would be
possible to represent it as a view). This includes any [tuple
index](multidimensional-indices/tuples.md) (i.e., multidimensional index) that
contains at least one array index.

```py
>>> a = np.arange(10)
>>> b = a[::4]
>>> c = a[[0, 4, 8]]
>>> b
array([0, 4, 8])
>>> c
array([0, 4, 8])
>>> b.base is a # b is a view of a
True
>>> print(c.base) # c is not (it is a copy)
None
```

Whether an array is a view or a copy matters for two reasons:

1. If you ever mutate the array, if it is a view, it will also mutate the data
   in the base array, as shown above. Be aware that views affect mutations in
   both directions: if `a` is a view, mutating it will also mutate whichever
   array it is a view on, but conversely, even if `a` is not a view, mutating
   it will modify any other arrays which are views into `a`. And while you can
   check if `a` is a view by looking at `a.base`, there is no easy way to
   check if `a` has other views pointing at it. The only way to know is to
   analyze the program and check any array which is created from `a`.

   It's best to minimize mutations in the presence of views, or to restrict
   them to a controlled part of the code, to avoid unexpected "[action at a
   distance](https://en.wikipedia.org/wiki/Action_at_a_distance_(computer_programming))"
   bugs.

   Note that you can always ensure that `a` is a new array that isn't a view
   and doesn't have any views pointing to it by copying it, using `a.copy()`.

2. Even if you don't mutate data, views are important because they are more
   efficient. A view is a relatively cheap thing to make, even if the array is
   large. It also saves on memory usage.

   For example, here we have an array with about 800 kB of data, and it takes
   over 200x longer to copy it than to create a view (using
   [IPython](https://ipython.org/)'s `%timeit`):

   ```
   In [1]: import numpy as np

   In [2]: a = np.arange(100000, dtype=np.float64)

   In [3]: %timeit a.copy()
   25.9 µs ± 645 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

   In [4]: %timeit a[:]
   127 ns ± 1.62 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
   ```

(strides)=
## Strides

The reason so many types of indexing into arrays can be represented as a
[view](views-vs-copies) without creating a copy is that NumPy arrays aren't
merely a pointer to a blob of memory. They are a pointer along with something
called *strides*. The strides tell NumPy how many bytes to skip in memory
along each axis to get to the next element of the array along that dimension.
This along with the *memory offset* (the address in physical memory of the
first byte of data), the *shape*, and the *itemsize* (the number of bytes each
element takes up, which depends on the *dtype*) exactly determines how the
corresponding memory is organized into an array. For example, let's start with
a flat 1-dimensional array with `24` elements whose itemsize is 8 (an `int64`
takes up 8 bytes). Its strides is `(8,)`:

```py
>>> a = np.arange(24)
>>> a.itemsize
8
>>> a.strides
(8,)
>>> a.shape
(24,)
```

Now let's create a view `b`, which is `a` reshaped to shape `(3, 2, 4)`. `b`
uses the exact same memory as `a` (which is just `0 1 2 ... 23`). Its itemsize
is the same because it has the same dtype, but its strides and shape are
different.

```py
>>> b = a.reshape((3, 2, 4))
>>> b
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> b.itemsize
8
>>> b.strides
(64, 32, 8)
>>> b.shape
(3, 2, 4)
```

This tells NumPy that to get the next element in the first dimension of `b`,
it needs to skip 64 bytes. That's because the first dimension contains 2\*4=8
items each, corresponding to the sizes of the second and third dimensions, and
each item is 8 bytes, so 8\*8=64. For example, the next element in the first
dimension after `0` (index `(0, 0, 0)`) is `8` (index `(1, 0, 0)`), which sits
exactly 64 bytes after it in memory. Similarly, to get the next element in the
second dimension, it should skip 32 bytes (4 elements).

The memory offset of an array can be accessed using `a.ctypes.data`. This is
the address in physical memory where the data (`0 1 2 ... 23`) lives. `a`
and `b` have the same memory offset because they both start with the same
first element:

```py
>>> a.ctypes.data # doctest: +SKIP
105553170825216
>>> b.ctypes.data # doctest: +SKIP
105553170825216
```

When we slice off the beginning of the array, we can see that all this does is
move the memory offset forward, and adjust the shape correspondingly.

```py
>>> a[2:].ctypes.data # doctest: +SKIP
105553170825232
>>> a[2:].shape
(22,)
>>> a[2:].strides # the strides are the same
(8,)
```

Specifically, it moves it by `2*8` (where remember `8` is `a.itemsize`):

```py
>>> a[2:].ctypes.data - a.ctypes.data
16
```

Here the strides are the same. Similarly, if we slice off the end of the
array, all it needs to do is adjust the shape. The memory offset is the same,
because it still starts at the same place in memory.

```py
>>> a[:2].ctypes.data # the memory offset is the same # doctest: +SKIP
105553170825216
>>> a[:2].shape
(2,)
>>> a[:2].strides # the strides are the same
(8,)
```

If we instead slice with a step, this adjusts the strides. For instance
`a[::2]` will double the strides, making it skip every other element (but the
offset will be again unchanged because it still starts at the first element of
`a`):

```py
>>> a[::2].strides
(16,)
>>> a[::2].ctypes.data # the memory offset is the same # doctest: +SKIP
105553170825216
>>> a[::2].shape
(12,)
```

If we use a *negative* step, the strides will become negative. This will cause
NumPy to work backwards in memory as it accesses the elements of the array.
The memory offset also changes here so that it starts with the last element of
`a`:

```py
>>> a[::-2].strides
(-16,)
>>> a[::-2].ctypes.data # doctest: +SKIP
105553170825400
```

From this, you are hopefully convinced that every possible slice is just a
manipulation of the memory offset, shape, and strides. It's not hard to see
that this also applies to integer indices (which just removes the stride for
the corresponding axis, adjusting the shape and memory offset accordingly) and
newaxis (which just adds `0` to the strides):

```py
>>> b.strides
(64, 32, 8)
>>> b[2].strides
(32, 8)
>>> b[np.newaxis].strides
(0, 64, 32, 8)
```

This is why [basic indexing](basic-indices) always produces a
[view](views-vs-copies): because it can always be represented as a
manipulation of the strides (plus shape and offset).

Another important fact about strides is that [broadcasting](broadcasting) can
be achieved by manipulating the strides, namely by using a `0` stride to
repeat the same data along a given axis.

```py
>>> c = a.reshape((1, 12, 2))
>>> d = np.broadcast_to(c, (5, 12, 2))
>>> c.shape
(1, 12, 2)
>>> c.strides
(192, 16, 8)
>>> d.shape
(5, 12, 2)
>>> d.strides
(0, 16, 8)
>>> c.base is a
True
>>> d.base is a
True
```

If you've ever used `broadcast_to()`, you might have noticed that it returns a
read-only array. That's because writing into it would not do what you'd
expect, since the repeated elements literally refer to the same memory.

This shows why [broadcasting](broadcasting) is so powerful: it can be done
without any actual copy of the data. When you perform an operation on two
arrays, the broadcasting is implicit, but even explicitly creating a
broadcasted array is cheap, because all it does is create a view with
different strides.

You can also manually create a view with any strides you want using [stride
tricks](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html).
Most views that you would want to create can be made with a combination of
indexing, `reshape`, `broadcast_to`, and `transpose`, but it's possible to use
strides to represent some things which are not so easy to do with just these
functions, for example, [sliding
windows](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html)
and [convolutions](https://stackoverflow.com/a/43087507/161801). However, if
you do use stride tricks, be careful of the caveats (see the [notes section of
the `as_strided`
docs](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)).

(c-vs-fortran-ordering)=
## C vs. Fortran Ordering

NumPy has an internal distinction between C order and Fortran
order.[^c-order-footnote] C-ordered arrays are stored in memory such that the
last axis varies the fastest. For example, if `a` has 3 dimensions, then its
elements are stored in memory like `a[0, 0, 0], a[0, 0, 1], a[0, 0, 2], ...,
a[0, 1, 0], a[0, 1, 1], a[0, 1, 2], ...`. Fortran ordering is the opposite:
the elements are stored in memory so that the first axis varies fastest, like
`a[0, 0, 0], a[1, 0, 0], a[2, 0, 0], ..., a[0, 1, 0], a[1, 1, 0], a[2, 1, 0]
...`.

[^c-order-footnote]: C ordering and Fortran ordering are also sometimes
  "row-major" and "column-major" ordering, respectively. However, this
  terminology is confusing when the array has more than two axes or when it
  does not represent a mathematical matrix. It's better to think of ordering
  in terms of which axes vary the fastest---the last for C ordering and the
  first for Fortran ordering. Also, I don't know about you, but I can never
  remember which is supposed to be "row-" and "column-" major, but I do
  remember how array indexing works in C, so just thinking about that requires
  no mnemonic.

The most important thing to note about C and Fortran ordering for our purposes
is that

> **the internal ordering of an array does not change any indexing
> semantics.**

The same index will select the same elements on `a` regardless of whether it
uses C or Fortran ordering internally.

More generally, the actual memory layout of an array has no bearing on
indexing semantics. Indexing operates on the logical abstraction of the array
as presented to the user, even if the true memory doesn't look anything like
that because the array is a [view](views-vs-copies) or has some other layout
due to [stride tricks](strides).

In particular, this also applies to [boolean array
indices](multidimensional-indices/boolean-arrays.md). A boolean mask always
[selects the elements in C order](boolean-array-c-order), even if the
underlying arrays use Fortran ordering.

```py
>>> a = np.arange(9).reshape((3, 3))
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> a_f = np.asarray(a, order='F')
>>> a_f # a_f looks the same as a, but the internal memory is ordered differently
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> idx = np.array([
...     [False,  True, False],
...     [ True,  True, False],
...     [False, False, False]])
>>> idx_f = np.asarray(idx, order='F') # It doesn't matter if the index array is Fortran-ordered either
>>> a[idx] # These are all the same
array([1, 3, 4])
>>> a_f[idx]
array([1, 3, 4])
>>> a[idx_f]
array([1, 3, 4])
>>> a_f[idx_f]
array([1, 3, 4])
```

~~~~{admonition} Aside
If you read the previous section on [strides](strides), you may have guessed
that the difference between C-ordered and Fortran-ordered arrays is a
difference of...strides!

```py
>>> a.strides
(24, 8)
>>> a_f.strides
(8, 24)
```

In a C-ordered array the strides decrease and in a Fortran-ordered array they
increase, because a smaller stride corresponds to "faster varying".
~~~~

**What ordering *does* affect is the performance of certain operations.** In
particular, the ordering determines whether it is more optimal to index along
the first or last axes of an array. For example, `a[0]` selects the first
subarray along the first axis (recall that `a[0]` is a [view](views-vs-copies)
into `a`, so it references the exact same memory as `a`). For a C-ordered
array, which is the default, this subarray is contiguous in memory. This is
because the indices on the last axes vary the fastest (i.e., are next to each
other in memory), so selecting a subarray of the first axis picks elements
which are still contiguous. Conversely, for a Fortran-ordered array, `a[0]` is
not contiguous, but `a[..., 0]` is.

```
>>> a[0].data.contiguous
True
>>> a_f[0].data.contiguous
False
>>> a_f[..., 0].data.contiguous
True
```

Operating on contiguous memory allows the CPU to place the entire memory block
in the cache at once, and is more performant as a result. The performance
difference won't be noticeable for our small example `a` above, as it is small
enough to fit entirely in the cache, but it becomes significant for larger
arrays. Compare the time to sum along `a[0]` or `a[..., 0]` for a
3-dimensional array `a` with a million elements using C and Fortran ordering
(using [IPython](https://ipython.org/)'s `%timeit`):

```
In [1]: import numpy as np

In [2]: a = np.ones((100, 100, 100)) # a has C order (the default)

In [3]: %timeit np.sum(a[0])
8.57 µs ± 121 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

In [4]: %timeit np.sum(a[..., 0])
24.2 µs ± 1.29 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [5]: a_f = np.asarray(a, order='F')

In [6]: %timeit np.sum(a_f[0])
26.3 µs ± 952 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [7]: %timeit np.sum(a_f[..., 0])
8.6 µs ± 130 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
```

Summing along contiguous memory (`a[0]` for C ordering and `a[..., 0]` for
Fortran ordering) is about 3x faster.

NumPy indexing semantics generally favor using C ordering, as it does not
require an ellipsis to select contiguous subarrays. C ordering also matches
the [list-of-lists intuition](multidimensional-indices/what-is-an-array.md) of
an array, since an array like `[[0, 1], [2, 3]]` is stored in memory as
literally `0, 1, 2, 3` with C ordering. It also aligns well with NumPy's
[broadcasting](broadcasting) rules, where broadcasted dimensions are prepended
by default, allowing one to think of an array as a "stack" of contiguous
subarrays.

C ordering is the default in NumPy when creating arrays with functions like
`asarray`, `ones`, `arange`, and so on. One typically only switches to
Fortran ordering when calling certain Fortran codes, or when creating an
array from another memory source that produces Fortran-ordered data.

Regardless of which ordering you are using, it is worth structuring your data
so that operations are done on contiguous memory when possible.

```{rubric} Footnotes
```
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
