# newaxis

The final basic multidimensional index type is `newaxis`. `np.newaxis` is an
alias for `None`. Both `np.newaxis` and `None` function identically; however,
`np.newaxis` is often more explicit than `None`, which may appear odd in an
index, so it is generally preferred. However, some people do use `None`
directly instead of `np.newaxis`, so it's important to remember that they are
the same thing.

```py
>>> import numpy as np
>>> print(np.newaxis)
None
>>> np.newaxis is None # They are exactly the same thing
True
```

`newaxis`, as the name suggests, adds a new axis to an array. This new axis
has size `1`. The new axis is added at the corresponding location within the
array shape. A size `1` axis neither adds nor removes any elements from the
array. Using the [nested lists analogy](what-is-an-array.md), it essentially
adds a new "layer" to the list of lists.


```py
>>> b = np.arange(4)
>>> b
array([0, 1, 2, 3])
>>> b[np.newaxis]
array([[0, 1, 2, 3]])
>>> b.shape
(4,)
>>> b[np.newaxis].shape
(1, 4)
```

Including `newaxis` alongside other indices in a [tuple index](tuples.md) does
not affect which axes those indices select. You can think of the `newaxis`
index as inserting the new axis in-place in the index, so that the other
indices still select the same corresponding axes they would select if it
weren't there.

Take our example array, which has shape `(3, 2, 4)`:

```py
>>> a = np.arange(24).reshape((3, 2, 4))
>>> a.shape
(3, 2, 4)
```

The index `a[0, :2]` results in a shape of `(2, 4)`: the integer index `0`
removes the first axis, the slice `:2` selects 2 elements from the second
axis, and the third axis is not selected at all, so it remains intact with 4
elements.

```py
>>> a[0, :2]
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])
>>> a[0, :2].shape
(2, 4)
```

Now, observe the shape of `a` when we insert `newaxis` at various points
within the index `a[0, :2]`:

```py
>>> a[np.newaxis, 0, :2].shape
(1, 2, 4)
>>> a[0, np.newaxis, :2].shape
(1, 2, 4)
>>> a[0, :2, np.newaxis].shape
(2, 1, 4)
>>> a[0, :2, ..., np.newaxis].shape
(2, 4, 1)
```

In each case, the exact same elements are selected: `0` always targets the
first axis, and `:2` always targets the second axis. The only difference is
where the size-1 axis is inserted:

```py
>>> a[np.newaxis, 0, :2]
array([[[0, 1, 2, 3],
        [4, 5, 6, 7]]])
>>> a[0, np.newaxis, :2]
array([[[0, 1, 2, 3],
        [4, 5, 6, 7]]])
>>> a[0, :2, np.newaxis]
array([[[0, 1, 2, 3]],
<BLANKLINE>
       [[4, 5, 6, 7]]])
>>> a[0, :2, ..., np.newaxis]
array([[[0],
        [1],
        [2],
        [3]],
<BLANKLINE>
       [[4],
        [5],
        [6],
        [7]]])
```

Let's look at each of these more closely:

1. `a[np.newaxis, 0, :2]`: the new axis is inserted before the first axis, but
the `0` and `:2` still index the original first and second axes. The resulting
shape is `(1, 2, 4)`.

2. `a[0, np.newaxis, :2]`: the new axis is inserted after the first axis, but
because the `0` removes this axis when it indexes it, the resulting shape is
still `(1, 2, 4)` (and the resulting array is the same).

3. `a[0, :2, np.newaxis]`: the new axis is inserted after the second axis,
because the `newaxis` comes right after the `:2`, which indexes the second
axis. The resulting shape is `(2, 1, 4)`. Remember that the `4` in the shape
corresponds to the last axis, which isn't represented in the index at all.
That's why in this example, the `4` still comes at the end of the resulting
shape.

4. `a[0, :2, ..., np.newaxis]`: the `newaxis` is after an ellipsis, so the new
axis is inserted at the end of the shape. The resulting shape is `(2, 4, 1)`.

In general, in a tuple index, the axis that each index selects corresponds to
its position in the tuple index after removing any `newaxis` indices.
Equivalently, `newaxis` indices can be though of as adding new axes *after*
the existing axes are indexed.

A size-1 axis can always be inserted anywhere in an array's shape without
changing the underlying elements.

An array index can include multiple instances of `newaxis` (or `None`). Each
will add a size-1 axis in the corresponding location.

**Exercise:** Can you determine the shape of this array, given that `a.shape`
is `(3, 2, 4)`?

```py
a[np.newaxis, 0, newaxis, :2, newaxis, ..., newaxis]
```

~~~~{dropdown} Click here to show the solution

```py
>>> a[np.newaxis, 0, np.newaxis, :2, np.newaxis, ..., np.newaxis].shape
(1, 1, 2, 1, 4, 1)
```

~~~~

In summary,

> **`np.newaxis` (which is just an alias for `None`) inserts a new size-1 axis
  in the corresponding location in the tuple index. The remaining,
  non-`newaxis` indices in the tuple index are indexed as if the `newaxis`
  indices were not there.**

(where-newaxis-is-used)=
## Where `newaxis` is Used

What we haven't said yet is why you would want to do such a thing in the first
place. One use case is to explicitly convert a 1-D vector into a 2-D matrix
representing a row or column vector. For example,

```py
>>> v = np.array([0, 1, -1])
>>> v.shape
(3,)
>>> v[np.newaxis]
array([[ 0,  1, -1]])
>>> v[np.newaxis].shape
(1, 3)
>>> v[..., np.newaxis]
array([[ 0],
       [ 1],
       [-1]])
>>> v[..., np.newaxis].shape
(3, 1)
```

`v[newaxis]` inserts an axis at the beginning of the shape, making `v` a `(1,
3)` row vector and `v[..., newaxis]` inserts an axis at the end, making it a
`(3, 1)` column vector.

But the most common usage is due to [broadcasting](broadcasting). The key idea
of broadcasting is that size-1 dimensions are not directly useful, in the
sense that they could be removed without actually changing anything about the
underlying data in the array. So they are used as a signal that that dimension
can be repeated in operations. `newaxis` is therefore useful for inserting
these size-1 dimensions in situations where you want to force your data to be
repeated. For example, suppose we have the two arrays

```py
>>> x = np.array([1, 2, 3])
>>> y = np.array([100, 200])
```

and suppose we want to compute an "outer" sum of `x` and `y`, that is, we want
to compute every combination of `a + b` where `a` is from `x` and `b` is from
`y`. The key realization here is that what we want is simply to
repeat each entry of `x` 2 times, to correspond to each entry of `y`, and
respectively repeat each entry of `y` 3 times, to correspond to each entry of
`x`. And this is exactly the sort of thing broadcasting does! We only need to
make the shapes of `x` and `y` match in such a way that the broadcasting will
do that. Since we want both `x` and `y` to be repeated, we will need to
broadcast both arrays. We want to compute

```py
[[ x[0] + y[0], x[0] + y[1] ],
 [ x[1] + y[0], x[1] + y[1] ],
 [ x[2] + y[0], x[2] + y[1] ]]
```

That way the first dimension of the resulting array will correspond to values
from `x`, and the second dimension will correspond to values from `y`, i.e.,
`a[i, j]` will be `x[i] + y[j]`. Thus the resulting array will have shape `(3,
2)`. So to make `x` (which is shape `(3,)`) and `y` (which is shape `(2,)`)
broadcast to this, we need to make them `(3, 1)` and `(1, 2)`, respectively.
This can easily be done with `np.newaxis`:

```py
>>> x[:, np.newaxis].shape
(3, 1)
>>> y[np.newaxis, :].shape
(1, 2)
```

Once we have the desired shapes, we just perform the operation, and NumPy will
do the broadcasting automatically.[^outer-footnote]

[^outer-footnote]: We could have also used the
    [`outer`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html)
    method of the `add` ufunc to achieve this, but using this for more a more
    complicated function than just `x + y` would be tedious, and it would not
    work in situations where you want to only repeat certain dimensions.
    Broadcasting is a more general way to do this, and `newaxis` is an
    important tool for making shapes align properly to make broadcasting do
    what you want.

```py
>>> x[:, np.newaxis] + y[np.newaxis, :]
array([[101, 201],
       [102, 202],
       [103, 203]])
```

Note: broadcasting automatically prepends size-1 dimensions, so the
`y[np.newaxis, :]` operation is unnecessary.

```py
>>> x[:, np.newaxis] + y
array([[101, 201],
       [102, 202],
       [103, 203]])
```

As we saw [before](single-axis-tuple), size-1 dimensions may seem redundant,
but they are not a bad thing. Not only do they allow indexing an array
uniformly, they are also very important in the way they interact with NumPy's
broadcasting rules.

```{rubric} Footnotes
```
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
