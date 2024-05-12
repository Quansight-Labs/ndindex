# Ellipses

Now that we understand how [tuple indices](tuples.md) work, the remaining
basic index types are relatively straightforward. The first type of index we
will look at is the ellipsis. An ellipsis is written as literally three dots:
`...`.[^ellipsis-footnote]

[^ellipsis-footnote]: You can also write out the word `Ellipsis`, but this is
    discouraged. In older versions of Python, the three dots `...` were not
    valid syntax outside of the square brackets of an index, but as of Python
    3, `...` is valid anywhere, making it unnecessary to use the spelled out
    `Ellipsis` in any context. The only reason I mention this is that if you
    type `...` at the interpreter, it will print "Ellipsis", and this explains
    why.

    ```py
    >>> ...
    Ellipsis
    ```

    This is also why the type name for the [ndindex `ellipsis`](ellipsis)
    object is lowercase, since `Ellipsis` is already a built-in name.

Consider an array with three dimensions:

```py
>>> import numpy as np
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

In [one of the examples in the previous section](tuples-slices-example), we
wanted to select only the first element of the last axis, and we saw that we
could use the index `:, :, 0`:

```py
>>> a[:, :, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

However, this index only works for our specific array, because it has 3
dimensions. If it had 5 dimensions instead, we would need to use `a[:, :, :,
:, 0]`. This is not only tedious to type, but also makes it impossible to
write an index that works for any number of dimensions. To contrast, if we
want the first element of the *first* axis, we could write `a[0]`, which
works if `a` has 3 dimensions or 5 dimensions or any number of dimensions.

The ellipsis solves this problem. An ellipsis index skips all the axes of an
array to the end, so that the indices after it select the last axes of the
array.

```py
>>> a[..., 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

You can also place indices before the ellipsis. The indices before the
ellipsis will select the first axes of the array, and the indices after it
will select the last axes. The ellipsis automatically skips all the
intermediate axes. For example, to select the first element of the first axis
and the last element of the last axis, we could use

```py
>>> a[0, ..., -1]
array([3, 7])
```

An ellipsis can also skip zero axes if all the axes of the array are already
accounted for. For example, these are the same because `a` has 3 dimensions:

```py
>>> a[1, 0:2, 2]
array([10, 14])
>>> a[1, 0:2, ..., 2]
array([10, 14])
```

Indeed, the index `1, 0:2, ..., 2` will work with any array that has *at
least* three dimensions (assuming of course that the first dimension is at
least size `2` and the last dimension is at least size `3`).

Previously, we saw that a [tuple index](tuples.md) implicitly ends in some
number of trivial `:` slices. We can also see here that a tuple index always
implicitly ends with an ellipsis, serving the same purpose. In other words:

> **An ellipsis automatically serves as a stand-in for the "correct" number of
trivial `:` slices to select the intermediate axes of an array**.

And just as with the
empty tuple index `()`, which we saw is the same as writing the right number
of trivial `:` slices, a single ellipsis and nothing else is the same as
selecting every axis of the array, i.e., it leaves the array
intact.[^tuple-ellipsis-footnote]

```py
>>> a[...]
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

Finally, only one ellipsis is allowed (otherwise it would be ambiguous which
axes are being indexed):

```py
>>> a[0, ..., 1, ..., 2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: an index can only have a single ellipsis ('...')
```

In summary, the rules for an ellipsis index are

- **An ellipsis index is written with three dots: `...`.**

- **`...` automatically selects 0 or more intermediate axes in an array.**

- **Every index before `...` operates on the first axes of the array. Every
  index after `...` operates on the last axes of the array.**

- **Every tuple index that does not have an ellipsis in it implicitly ends in
  `...`.**

- **At most one `...` is allowed in a tuple index.**

```{rubric} Footnotes
```
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
