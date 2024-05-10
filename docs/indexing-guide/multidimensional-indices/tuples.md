(tuple-indices)=
# Tuples

The basic building block of multidimensional indexing is the `tuple` index. A
tuple index doesn't select elements on its own. Instead, it contains other
indices that themselves select elements. The general rule for tuples is that

> **each element of a tuple index selects the corresponding elements for the
  corresponding axis of the array**

(this rule is modified a little bit in the presence of ellipses or newaxis, as
we will see below).

For example, suppose we have a three-dimensional array `a` with the
shape `(3, 2, 4)`. For simplicity, we'll define `a` as a reshaped `arange`, so
that each element is distinct and we can easily see which elements are
selected.

```py
>>> a = np.arange(24).reshape((3, 2, 4))
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

If we use a basic single axis index on `a` such as an integer or slice, it
will operate on the first dimension of `a`:

```py
>>> a[0] # The first row of the first axis
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])
>>> a[2:] # The elements that are not in the first or second rows of the first axis
array([[[16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

We also observe that integer indices remove the axis, and slices keep the axis
(even when the resulting axis has size-1):

```py
>>> a[0].shape
(2, 4)
>>> a[2:].shape
(1, 2, 4)
```

The indices in a tuple index target the corresponding elements of the
corresponding axis. So for example, the index `(1, 0, 2)` selects the second
element of the first axis, the first element of the second axis, and the third
element of the third axis (remember that indexing is 0-based, so index `0`
corresponds to the first element, index `1` to the second, and so on). Looking
at the list of lists representation of `a` that was printed by NumPy:

```py
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

The first index is `1`, so we should take the second element of the outermost list, giving

```py
[[ 8,  9, 10, 11],
 [12, 13, 14, 15]]

```

The next index is `0`, so we get the first element of this list, which is the list


```py
[ 8,  9, 10, 11]
```

Finally, the last index is `2`, giving the third element of this list:

```py
10
```

And indeed:

```py
>>> a[(1, 0, 2)] # doctest: +SKIPNP1
np.int64(10)
```

If we had stopped at an intermediate tuple, instead of getting an element, we
would have gotten the subarray that we accessed. For example, just `(1,)`
gives us the first intermediate array we looked at:

```py
>>> a[(1,)]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

And `(1, 0)` gives us the second intermediate array we looked at:

```py
>>> a[(1, 0)]
array([ 8,  9, 10, 11])
```

In each case, the integers remove the corresponding axes from the array shape:

```py
>>> a.shape
(3, 2, 4)
>>> a[(1,)].shape
(2, 4)
>>> a[(1, 0)].shape
(4,)
```

We can actually think of the final element, `10`, as being an array with shape
`()` (0 dimensions). Indeed, NumPy agrees with this idea:

```py
>>> a[(1, 0, 2)].shape
()
```

Now, it's important to note a key point about tuple indices: **the parentheses
in a tuple index are completely optional.** Instead of writing `a[(1, 0, 2)]`,
we could simply write `a[1, 0, 2]`.

```py
>>> a[1, 0, 2] # doctest: +SKIPNP1
np.int64(10)
```

These are exactly the same. When the parentheses are omitted, Python
automatically treats the index as a tuple. From here on, we will always omit
the parentheses, as is common practice. Not only is this cleaner, but it is
also important for another reason: syntactically, Python does not allow slices
in a tuple index if the parentheses are included:


```py
>>> a[(1:, :, :-1)] # doctest: +SKIP
  File "<stdin>", line 1
    a[(1:, :, :-1)]
        ^
SyntaxError: invalid syntax
>>> a[1:, :, :-1]
array([[[ 8,  9, 10],
        [12, 13, 14]],
<BLANKLINE>
       [[16, 17, 18],
        [20, 21, 22]]])
```

Now, let's go back and look at an example we just showed:

```py
>>> a[(1,)] # or just a[1,]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

You might have noticed something about this. It is selecting the second element
of the first axis. But from what we said earlier, we can also do this just by
using the basic index `1`, which will operate on the first axis:

```py
>>> a[1] # Exactly the same thing as a[(1,)]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

This illustrates the first important fact about tuple indices:

> **A tuple index with a single index, `a[i,]`, is exactly the same as that
  single index, `a[i]`.**

The reason is that in both cases, the index `i` operates over the
first axis of the array. This is true no matter what kind of index `i` is. `i`
can be an integer index, a slice, an ellipsis, and so on. With one exception,
that is: `i` cannot itself be a tuple index! Nested tuple indices are not
allowed.

In practice, this means that when working with NumPy arrays, you can think of
every index type as a single element tuple index. An integer index `0` is
"actually" the tuple index `(0,)`. The slice `a[0:3]` is actually a tuple
`a[0:3,]`. This is a good way to think about indices because it will help you
remember that non-tuple indices operate as if they were the first element of a
single-element tuple index, namely, they operate on the first axis of the
array. Remember, however, that this does not apply to Python built-in types;
for example, `l[0,]` and `l[0:3,]` will both produce errors if `l` is a
`list`, `tuple`, or `str`.

Up to now, we looked at the tuple index `(1, 0, 2)`, which selected a single
element. And we considered sub-tuples of this, `(1,)` and `(1, 0)`, which
selected subarrays. What if we want to select other subarrays? For example,
`a[1, 0]` selects the subarray with the second element of the first axis and
the first element of the second axis. What if instead we wanted the first
element of the *last* axis (axis 3)?

We can do this with slices. In particular, the trivial slice `:` will select
every single element of an axis (remember that the `:` slice means ["select
everything"](omitted)). So we want to select every element from the first and
second axis, and only the first element of the last axis, meaning our index is
`:, :, 0`:

```py
>>> a[:, :, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

`:` serves as a convenient way to "skip" axes. It is one of the most common
types of indices that you will see in practice for this reason. However, it is
important to remember that `:` is not special. It is just a slice, which selects
every element of the corresponding axis. We could also replace `:` with `0:n`,
where `n` is the size of the corresponding axis.

```py
>>> a[0:3, 0:2, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

Of course, in practice using `:` is better because we might not know or care
what the actual size of the axis is, and it's less typing anyway.

When we used the indices `(1,)` and `(1, 0)`, we observed that they targeted
the first and the first two axes, respectively, leaving the remaining axes
intact and producing subarrays. Another way of saying this is that the each
tuple index implicitly ended with `:` slices, one for each axis we didn't
index:

```py
>>> a[1,]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> a[1, :, :]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> a[1, 0]
array([ 8,  9, 10, 11])
>>> a[1, 0, :]
array([ 8,  9, 10, 11])
```

This is a rule in general:

> **A tuple index implicitly ends in as many slices `:` as there are remaining
  dimensions of the array.**

(single-axis-tuple)=
The [slices](slices-docs) page stressed the point that [slices always keep the
axis they index](subarray), but it wasn't clear why that is important until
now. Suppose we slice the first axis of `a`, then later, we take that array
and want to get the first element of the last row.


```py
>>> n = 2
>>> b = a[:n]
>>> b[-1, -1, 0] # doctest: +SKIPNP1
np.int64(12)
```

Here `b = a[:2]` has shape `(2, 2, 4)`

```
>>> b.shape
(2, 2, 4)
```

But suppose we used a slice that only selected one element from the first axis
instead

```py
>>> n = 1
>>> b = a[:n]
>>> b[-1, -1, 0] # doctest: +SKIPNP1
np.int64(4)
```

It still works. Here `b` has shape `(1, 2, 4)`:

```py
>>> b.shape
(1, 2, 4)
>>> b
array([[[0, 1, 2, 3],
        [4, 5, 6, 7]]])
```

Even though the slice `a[:1]` only produces a single element in the first
axis, that axis is maintained as size `1`. We might think this array is
"equivalent" to the same array with shape `(2, 4)`, since the first axis is
redundant (the outermost list only has one element, so we don't really need
it).

```py
>>> # c is kind of the same as b above
>>> c = np.array([[0, 1, 2, 3],
...               [4, 5, 6, 7]])
```

This is true in the sense that the elements are the same, but the
resulting array has different properties. Namely, the index we used for `b`
will not work for it.

```py
>>> c[-1, -1, 0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: too many indices for array: array is 2-dimensional, but 3 were
indexed
```

Here we tried to use the same index on `c` that we used on `b`, but it didn't
work, because our index assumed three axes, but `c` only has two:

```py
>>> c.shape
(2, 4)
```

Thus, when it comes to indexing, all axes, even "trivial" axes, matter. It's
sometimes a good idea to maintain the same number of dimensions in an array
throughout a computation, even if one of them sometimes has size 1, simply
because it means that you can index the array
uniformly.[^size-1-dimension-footnote] And this doesn't apply just to
indexing. Many NumPy functions reduce the number of dimensions of their output
(for example, {external+numpy:func}`numpy.sum`), but they have a `keepdims`
argument to retain the dimension as a size-1 dimension instead.

[^size-1-dimension-footnote]: In this example, if we knew that we were always
    going to select exactly one element (say, the second one) from the first
    dimension, we could equivalently use `a[1, np.newaxis]` (see
    [](../integer-indices) and [](newaxis-indices)). The advantage of this is
    that we would get an error if the first dimension of `a` didn't actually
    have `2` elements, whereas `a[1:2]` would just silently give a size-0
    array.

There are two final facts about tuple indices that should be noted before we
move on to the other basic index types. First, as we noticed above,

> **if a tuple index has more elements than there are dimensions in an array,
  it raises an `IndexError`.**

Secondly, an array can be indexed by an empty tuple `()`. If we think about it
for a moment, we said that every tuple index implicitly ends in enough trivial
`:` slices to select the remaining axes of an array. That means that for an
array `a` with $n$ dimensions, an empty tuple index `a[()]` should be the same
as `a[:, :, … (n times)]`. This would select every element of every axis. In
other words,

> **the empty tuple index `a[()]` always just returns the entire array `a`
  unchanged.**[^tuple-ellipsis-footnote]

[^tuple-ellipsis-footnote]: There is one important distinction between the
    empty tuple index (`a[()]`) and a single ellipsis index (`a[...]`). NumPy
    makes a distinction between scalars and 0-D (i.e., shape `()`) arrays. On
    either, an empty tuple index `()` will always produce a scalar, and a
    single ellipsis `...` will always produce a 0-D array:

    ```py
    >>> s = np.int64(0) # scalar
    >>> x = np.array(0) # 0-D array
    >>> s[()] # doctest: +SKIPNP1
    np.int64(0)
    >>> x[()] # doctest: +SKIPNP1
    np.int64(0)
    >>> s[...]
    array(0)
    >>> x[...]
    array(0)
    ```

    This also applies for tuple indices that select a single element. If the
    tuple contains a (necessarily redundant) ellipsis, the result is a 0-D
    array. Otherwise, the result is a scalar. With the example array:

    ```py
    >>> a[1, 0, 2] # scalar # doctest: +SKIPNP1
    np.int64(10)
    >>> a[1, 0, 2, ...] # 0-D array
    array(10)
    ```

    The difference between scalars and 0-D arrays in NumPy is subtle. In most
    contexts, they will both work identically, but, rarely, you may need one
    and not the other, and the above trick can be used to convert between
    them. See footnotes [^integer-scalar-footnote] and [1 in "Other Topics Relevant
    to Indexing"](view-scalar-footnote-ref) for two important
    differences between the scalars and 0-D arrays which are related to indexing.

## Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
