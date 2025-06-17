# Boolean Array Indices

The final index type is boolean arrays. Boolean array indices are also
sometimes called *masks*,[^mask-footnote] because they "mask out" elements of
the array.

```{note}
In this section, as with [the previous](integer-arrays.md), do not confuse the
*array being indexed* with the *array that is the index*. The former can be
anything and have any dtype. It is only the latter that is restricted to being
integer or boolean.
```

[^mask-footnote]: Not to be confused with {external+numpy:std:doc}`NumPy
    masked arrays <reference/maskedarray>`.

A boolean array index specifies which elements of an array should be selected
and which should not be selected.

The simplest and most common case is where a boolean array index has the same
shape as the array being indexed, and is the sole index (i.e., not part of a
larger [tuple index](tuples.md)).

Consider the array:

```py
>>> import numpy as np
>>> a = np.arange(9).reshape((3, 3))
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
```

Suppose we want to select the elements `1`, `3`, and `4`: to do so, we create a
boolean array of the same shape as `a` which is `True` in the positions where
those elements are and `False` everywhere else.

```py
>>> idx = np.array([
...     [False,  True, False],
...     [ True,  True, False],
...     [False, False, False]])
>>> a[idx]
array([1, 3, 4])
```

From this we can see a few things:

- The result of indexing with the boolean mask is a 1-D array. If we think
  about it, this is the only possibility. A boolean index could select any
  number of elements. In this case, it selected 3 elements, but it could
  select as few as 0 and as many as 9 elements from `a`. So there would be no
  way to return a higher dimensional shape or for the shape of the result to
  be somehow related to the shape of `a`.

- The selected elements are "in order" ([more on what this means
  later](boolean-array-c-order)).

However, these details are usually not important. This is because an array
indexed by a boolean array is typically used indirectly, such as on the
left-hand side of an assignment.

A typical use case of boolean indexing involves creating a boolean mask using
the array itself with operators that return boolean arrays, such as relational
operators (`<`, `<=`, `==`, `>`, `>=`, `!=`), logical operators (`&` (and),
`|` (or), `~` (not), `^` (xor)), and boolean functions (e.g.,
{external+numpy:py:data}`isnan() <numpy.isnan>` or
{external+numpy:py:data}`isinf() <numpy.isinf>`).

Consider an array of the integers from -10 to 10:

```py
>>> a = np.arange(-10, 11)
>>> a
array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,
         3,   4,   5,   6,   7,   8,   9,  10])
```

Say we want to select the elements of `a` that are both positive and odd. The
boolean array `a > 0` represents which elements are positive and the boolean
array `a % 2 == 1` represents which elements are odd. So our mask would be

```py
>>> mask = (a > 0) & (a % 2 == 1)
```

Note the careful use of parentheses to match [Python operator
precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence).
Masks must use the logical operators `&`, `|`, and `~` so that they can
operate on arrays. They cannot use the Python keywords `and`, `or`, and `not`,
because they don't work on arrays.

Our `mask` is just an array of booleans:

```py
>>> mask
array([False, False, False, False, False, False, False, False, False,
       False, False,  True, False,  True, False,  True, False,  True,
       False,  True, False])
```

To get the actual matching elements, we need to index `a` with the mask:

```py
>>> a[mask]
array([1, 3, 5, 7, 9])
```

Often, one will see the `mask` written directly in the index, like

```py
>>> a[(a > 0) & (a % 2 == 1)]
array([1, 3, 5, 7, 9])
```

Suppose we want to set these elements of `a` to `-100` (i.e., to "mask" them
out). This can be done easily with an indexing
assignment[^indexing-assignment-footnote]:

[^indexing-assignment-footnote]: All the indexing rules discussed in this
    guide apply when the indexed array is on the left-hand side of an `=`
    assignment. The elements of the array that are selected by the index are
    assigned in-place to the array or number on the right-hand side.

```
>>> a[(a > 0) & (a % 2 == 1)] = -100
>>> a
array([ -10,   -9,   -8,   -7,   -6,   -5,   -4,   -3,   -2,   -1,    0,
       -100,    2, -100,    4, -100,    6, -100,    8, -100,   10])
```

One common use case of this sort of thing is to mask out `nan` entries with a
finite number, like `0`:

```
>>> a = np.linspace(-5, 5, 10)
>>> b = np.log(a)
>>> b
array([        nan,         nan,         nan,         nan,         nan,
       -0.58778666,  0.51082562,  1.02165125,  1.35812348,  1.60943791])
>>> b[np.isnan(b)] = 0.
>>> b
array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.58778666,  0.51082562,  1.02165125,  1.35812348,  1.60943791])
```

Here `np.isnan(x)` returns a boolean array of the same shape as `x` that is
`True` if the corresponding element is `nan` and `False` otherwise.

Note that for this kind of use case, the actual shape of `a[mask]` is
irrelevant. The important thing is that it is some subset of `a`, which is
then assigned to, mutating only those elements of `a`.

It's important to not be fooled by this way of constructing a mask. Even
though the *expression* `(a > 0) & (a % 2 == 1)` depends on `a`, the resulting
*array itself* does not---it is just an array of booleans. **Boolean array
indexing, as with [all other types of indexing](../intro.md), does not depend
on the values of the array, only on the positions of its elements.**

This distinction might feel overly pedantic, but it matters once you realize
that a mask created with one array can be used on another array, so long as it
has the same shape. It is common to have multiple arrays representing
different data about the same set of points. You may want to select a subset
of one array based on the values of the corresponding points in another array.

For example, suppose we want to plot the function $f(x) = 4x\sin(x) -
\frac{x^2}{4} - 2x$ on $[-10,10]$. We can set `x = np.linspace(-10, 10)` and
compute the array expression:

<!-- myst doesn't work with ```{plot}, and furthermore, if the two plot
directives are put in separate eval-rst blocks, the same plot is copied to
both. -->

```{eval-rst}
.. plot::
   :context: reset
   :include-source: True
   :filename-prefix: function-plot
   :alt: A plot of 4*x*np.sin(x) - x**2/4 - 2*x from -10 to 10. The curve crosses the x-axis several times at irregular intervals.
   :caption: Plot of :math:`y = 4x\sin(x) - \frac{x^2}{4} - 2x`

   >>> import matplotlib.pyplot as plt
   >>> x = np.linspace(-10, 10, 10000) # 10000 evenly spaced points between -10 and 10
   >>> y = 4*x*np.sin(x) - x**2/4 - 2*x # our function
   >>> plt.scatter(x, y, marker=',', s=1)
   <matplotlib.collections.PathCollection object at ...>

If we want to show only those :math:`x` values that are positive, we could
easily do this by modifying the ``linspace`` call that created ``x``. But what
if we want to show only those :math:`y` values that are positive? The only way
to do this is to select them using a mask:

.. plot::
   :context: close-figs
   :include-source: True
   :filename-prefix: function-plot-masked
   :alt: A plot of only the parts of 4*x*np.sin(x) - x**2/4 - 2*x that are above the x-axis.
   :caption: Plot of :math:`y = 4x\sin(x) - \frac{x^2}{4} - 2x` where :math:`y > 0`

   >>> plt.scatter(x[y > 0], y[y > 0], marker=',', s=1)
   <matplotlib.collections.PathCollection object at ...>

```

Here we are using the mask `y > 0` to select the corresponding values from
*both* the `x` and the `y` arrays. Since the same mask is used on both arrays,
the values corresponding to this mask in both arrays will be selected. With
`x[y > 0]`, even though the mask itself is not strictly created *from* `x`, it
still makes sense as a mask for the array `x`. In this case, the mask selects
a nontrivial subset of `x`.

Using a boolean array mask created from a different array is very common. For
example, in [scikit-image](https://scikit-image.org/), an image is represented
as an array of pixel values. Masks can be used to select a subset of the
image. A mask based on the pixel values (e.g., all red pixels) would depend on
the array, but a mask based on a geometric shape independent of the pixel
values, such as a
[circle](https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_camera_numpy.html),
would not. In that case, the mask would just be a circular arrangement of
`True`s and `False`s. As another example, in machine learning, if `group` is
an array with group numbers and `X` is an array of features with repeated
measurements per group, one can select the features for a single group to do
cross-validation like `X[group == 0]`.

## Advanced Notes

As [with integer array indices](integer-arrays-advanced-notes), the above
section provides the basic gist of boolean array indexing, but there are some
advanced semantics described below, which can be skipped by new NumPy users.

(boolean-array-result-shape)=
### Result Shape

> **A boolean array index will remove as many dimensions as the index has, and
> replace them with a single flat dimension, which has size equal to the
> number of `True` elements in the index.**

The shape of the boolean array index must exactly match the dimensions being
replaced, or the index will result in an `IndexError`.

For example:

```py
>>> a = np.arange(24).reshape((2, 3, 4))
>>> idx = np.array([[True, False, True],
...                 [True, True, True]])
>>> a.shape
(2, 3, 4)
>>> idx.shape # Matches the first two dimensions of a
(2, 3)
>>> np.count_nonzero(idx) # The number of True elements in idx
5
>>> a[idx].shape # The (2, 3) in a.shape is replaced with count_nonzero(idx)
(5, 4)
```

This means that the final shape of an array indexed with a boolean mask
depends on the value of the mask, specifically, the number of `True` values in
it. It is easy to construct array expressions with boolean masks where the
size of the array cannot be determined until runtime. For example:

```py
>>> rng = np.random.default_rng(11) # Seeded so this example reproduces
>>> a = rng.integers(0, 2, (3, 4)) # A shape (3, 4) array of 0s and 1s
>>> a[a==0].shape # Could be any size from 0 to 12
(7,)
```

However, even if the number of elements in an indexed array is not
determinable until runtime, the *number of dimensions* is determinable. This
is because a boolean mask acts as a flattening operation. All the dimensions
of the boolean array index are removed from the indexed array and replaced
with a single dimension. Only the *size* of this dimension cannot be
determined, unless the number of `True` elements in the index is known.

This detail means that sometimes code that uses boolean array indexing can be
difficult to reason about statically, because the array shapes are inherently
unknowable until runtime and may depend on data. For this reason, array
libraries that build computational graphs from array expressions without
evaluating them, such as
[JAX](https://jax.readthedocs.io/en/latest/index.html) or [Dask
Array](https://docs.dask.org/en/stable/array.html), may have limited or no
support for boolean array indexing.

(boolean-array-c-order)=
### Result Order

> **The order of the elements selected by a boolean array index `idx`
> corresponds to the elements being iterated in C order.**

C order iterates the array `a` so that the last axis varies the fastest,
like `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`, `(0, 1, 0)`, `(0, 1, 1)`, etc.

For example:

```py
>>> a = np.arange(12).reshape((3, 4))
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> idx = np.array([[ True, False,  True,  True],
...                 [False,  True, False, False],
...                 [ True,  True, False,  True]])
>>> a[idx]
array([ 0,  2,  3,  5,  8,  9, 11])
```

In this example, the elements of `a` are ordered `0 1 2 ...` in C order, which
is why in the final indexed array `a[idx]`, they are still in sorted order. C
order also corresponds to reading the elements of the array in the order that
NumPy prints them, from left to right, ignoring the brackets and commas.

C ordering is always used, even when the underlying memory is not C-ordered
(see [](c-vs-fortran-ordering) for more details on C array ordering).

### Masking a Subset of Dimensions

It is possible to use a boolean mask to select only a subset of the dimensions
of `a`. For example, let's take a shape `(2, 3, 4)` array `a`:

```py
>>> a = np.arange(24).reshape((2, 3, 4))
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
<BLANKLINE>
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

Say we want to select the elements of `a` that are greater than 5, but only in
the first subarray along the first dimension (only the elements from 0 to 11).
We can create a mask on only that subarray:

```py
>>> mask = a[0] > 5
>>> mask.shape
(3, 4)
```

Then, apply it to that same subarray:

```py
>>> a[0, mask]
array([ 6,  7,  8,  9, 10, 11])
```

The [tuple](tuples.md) index `(0, mask)` works just like any other tuple
index: it selects the subarray `a[0]` along the first axis, then applies the
`mask` to the remaining dimensions. The shape of `mask`, `(3, 4)`, matches
those remaining dimensions (by construction), so the index is valid.

Masking a subset of dimension is not as common as masking the entire array
`a`, but it does happen. Remember that we can always think of an array as an
"array of subarrays". For instance, suppose we have a video with 1920 x 1080
pixels and 500 frames. This might be represented as an array of shape `(500,
1080, 1920, 3)`, where the final dimension, 3, represents the 3 RGB color
values of a pixel. We can think of this array as 500 different 1080 &times;
1920 &times; 3 "frames". Or as a 500 &times; 1080 &times; 1920 array of
3-tuple "pixels". Or we could slice along the last dimension and think of it
as three 500 &times; 1080 &times; 1920 video "channels", one for each primary
color.

In each case, we imagine that our array is really an array (or a stack or
batch) of subarrays, where some of our dimensions are the "stacking"
dimensions and some of them are the array dimensions. This way of thinking is
also common when doing linear algebra on arrays. The last two dimensions
(typically) are considered matrices, and the leading dimensions are batch
dimensions. An array of shape `(10, 5, 4)` might be thought of as ten 5
&times; 4 matrices. NumPy linear algebra functions like `solve` and the `@`
matmul operator will automatically operate on the last two dimensions of an
array.

So, how does this relate to using a boolean array index to select only a
subset of the array dimensions? Well, we might want to use a boolean index to
select only along the inner "subarray" dimensions, and pretend like the outer
"batching" dimensions are our "array".

For example, say we have an image represented in
[scikit-image](https://scikit-image.org/) as a 3-D array:

```{eval-rst}
.. plot::
   :context: reset
   :include-source: True
   :filename-prefix: astronaut
   :alt: An image of an astronaut, which is represented as a shape (512, 512, 3) array.

   >>> def imshow(image, title):
   ...     import matplotlib.pyplot as plt
   ...     plt.axis('off')
   ...     plt.title(title)
   ...     plt.imshow(image)
   >>> from skimage.data import astronaut
   >>> image = astronaut()
   >>> image.shape
   (512, 512, 3)
   >>> imshow(image, "Original Image")

Now, suppose we want to increase the saturation of this image. We can do this
by converting the image to `HSV space
<https://en.wikipedia.org/wiki/HSL_and_HSV>`_ and increasing the saturation
value (the second value in the last dimension, which should always be between
0 and 1):

.. plot::
   :context: close-figs
   :include-source: True
   :filename-prefix: astronaut-saturated-
   :alt: An image of an astronaut with increased saturation. The lighter parts of the image appear washed out.

   >>> from skimage import color
   >>> hsv_image = color.rgb2hsv(image)
   >>> # Add 0.3 to the saturation, clipping the values to the range [0, 1]
   >>> hsv_image[..., 1] = np.clip(hsv_image[..., 1] + 0.3, 0, 1)
   >>> # Convert back to RGB
   >>> saturated_image = color.hsv2rgb(hsv_image)
   >>> imshow(saturated_image, "Saturated Image (Naive)")

However, this ends up looking bad and washed out, because the whole image now
has a minimum saturation of 0.3. A better approach would be to select the
pixels that already have a saturation above some threshold, and increase the
saturation of only those pixels:

.. plot::
   :context: close-figs
   :include-source: True
   :filename-prefix: astronaut-saturated-better
   :alt: An image of an astronaut with increased saturation. The image does not appear washed out.

   >>> hsv_image = color.rgb2hsv(image)
   >>> # Mask only those pixels whose saturation is > 0.6
   >>> high_sat_mask = hsv_image[:, :, 1] > 0.6
   >>> # Increase the saturation of those pixels by 0.3
   >>> hsv_image[high_sat_mask, 1] = np.clip(hsv_image[high_sat_mask, 1] + 0.3, 0, 1)
   >>> # Convert back to RGB
   >>> enhanced_color_image = color.hsv2rgb(hsv_image)
   >>> imshow(enhanced_color_image, "Saturated Image (Better)")

```

Here, `hsv_image.shape` is `(512, 512, 3)`, so our mask `hsv_image[:, :, 1] >
0.6`[^high_sat_mask-footnote] has shape `(512, 512)`, i.e., the shape of the
first two dimensions. In other words, the mask has one value for each pixel,
either `True` if the saturation is `> 0.6` or `False` if it isn't. To add
`0.3` saturation to only those pixels above the threshold, we mask the
original array with `hsv_image[high_sat_mask, 1]`. The `high_sat_mask` part of
the index selects only those pixel values that have high saturation, and the
`1` in the final dimension selects the saturation channel for those pixels.

[^high_sat_mask-footnote]: We could have also written `(hsv_image > 0.6)[:, :,
    1]`, although this would be less efficient because it would unnecessarily
    compute `> 0.6` for the hue and value channels.

(nonzero-equivalence)=
### `nonzero()` Equivalence

Another way to think about boolean array indices is based on the
`np.nonzero()` function. `np.nonzero(x)` returns a tuple of arrays of integer
indices where `x` is nonzero, or in the case where `x` is boolean, where `x`
is True. For example:

```py
>>> idx = np.array([[ True, False,  True,  True],
...                 [False,  True, False, False],
...                 [ True,  True, False,  True]])
>>> np.nonzero(idx)
(array([0, 0, 0, 1, 2, 2, 2]), array([0, 2, 3, 1, 0, 1, 3]))
```

The first array in the tuple corresponds to indices for the first dimension;
the second array to the second dimension, and so on. If this seems familiar,
it's because this is exactly how we saw that [multidimensional integer array
indices](multidimensional-integer-indices) worked. Indeed, there is a basic
equivalence between the two:

> **A boolean array index `idx` is the same as if you replaced `idx` with the
result of {external+numpy:func}`np.nonzero(idx) <numpy.nonzero>` (unpacking
the tuple), using the rules for [integer array indices](integer-arrays.md)
outlined previously.**

Note, however, that this rule *does not* apply to [0-dimensional boolean
indices](0-d-boolean-index).

```py
>>> a = np.arange(12).reshape((3, 4))
>>> a[idx]
array([ 0,  2,  3,  5,  8,  9, 11])
>>> np.nonzero(idx)
(array([0, 0, 0, 1, 2, 2, 2]), array([0, 2, 3, 1, 0, 1, 3]))
>>> idx0, idx1 = np.nonzero(idx)
>>> a[idx0, idx1] # this is the same as a[idx]
array([ 0,  2,  3,  5,  8,  9, 11])
```

Here `np.nonzero(idx)` returns two integer array indices, one for each
dimension of `idx`. These indices each have `7` elements, one for each
`True` element of `idx`, and they select (in C order), the corresponding
elements. Another way to think of this is that `idx[np.nonzero(idx)]` will
always return an array of `np.count_nonzero(idx)` `True`s, because
`np.nonzero(idx)` is exactly the integer array indices that select the
`True` elements of `idx`:

```py
>>> idx[np.nonzero(idx)]
array([ True,  True,  True,  True,  True,  True,  True])
```

What this all means is that all the rules that are outlined previously about
[integer array indices](integer-arrays.md), e.g., [how they
broadcast](integer-array-broadcasting) or [combine together with
slices](integer-arrays-combined-with-basic-indices), all also apply to boolean
array indices after this transformation. This also specifies how boolean array
indices and integer array indices combine
together.[^combining-integer-and-boolean-indices-footnote]

[^combining-integer-and-boolean-indices-footnote]: Combining an integer array
    and boolean array index together is not common, as the shape of the
    integer array index would have to be broadcast compatible with the number
    of `True` elements in the boolean array.

    ```py
    >>> a = np.arange(10).reshape((2, 5))
    >>> a[np.array([0, 1, 0]), np.array([True, False, True, True, False])]
    array([0, 7, 3])
    >>> a[np.array([0, 1, 0]), np.array([True, False, True, True, True])]
    Traceback (most recent call last):
    ...
    IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (4,)
    ```

    It's not impossible for this to come up in practice, but like many of the
    advanced indexing semantics discussed here, it's mostly supported for the
    sake of completeness.

Effectively, a boolean array index can be combined with other boolean or
integer array indices by first converting the boolean index into integer
indices (one for each dimension of the boolean index) that select each `True`
element of the index, and then broadcasting them all to a common shape.

The ndindex method
[`Tuple.broadcast_arrays()`](ndindex.Tuple.broadcast_arrays) (as well as
[`expand()`](ndindex.Tuple.expand)) will convert boolean array indices into
integer array indices via {external+numpy:func}`numpy.nonzero` and broadcast
array indices together into a canonical form.

(0-d-boolean-index)=
### Boolean Scalar Indices

A 0-dimensional boolean index (i.e., just the scalar `True` or `False`) is a
little special. The [`np.nonzero` rule](nonzero-equivalence) stated above does
not actually apply. This is because `np.nonzero` exhibits odd behavior with
0-D arrays. `np.nonzero(a)` usually returns a tuple with as many arrays as
dimensions of `a`:

```py
>>> np.nonzero(np.array([True, False]))
(array([0]),)
>>> np.nonzero(np.array([[True, False]]))
(array([0]), array([0]))
```

But for a 0-D array, `np.nonzero(a)` doesn't return an empty tuple, but
rather the same thing as
`np.nonzero(np.array([a]))`:[^nonzero-deprecated-footnote]

<!-- TODO: Update this text when NumPy 2.0 is released. -->
[^nonzero-deprecated-footnote]: In NumPy 2.0, calling `nonzero()` on a 0-D
    array is deprecated, and in NumPy 2.1 it will result in an error,
    precisely due to this odd behavior.


```py
>>> np.nonzero(np.array(False)) # doctest: +SKIP
(array([], dtype=int64),)
>>> np.nonzero(np.array(True)) # doctest: +SKIP
(array([0]),)
```

However, the key point---that a [boolean array index removes `idx.ndim`
dimensions from `a` and replaces them with a single dimension with size equal
to the number of `True` elements](boolean-array-result-shape)---remains true.
Here, `idx.ndim` is `0`, because `array(True)` and `array(False)` have shape
`()`. Thus, these indices "remove" 0 dimensions and add a single dimension of
size 1 for `True` or 0 for `False`. Hence, if `a` has shape `(s1, ..., sn)`,
then `a[True]` has shape `(1, s1, ..., sn)`, and `a[False]` has shape `(0, s1,
..., sn)`.

```py
>>> a.shape # as above
(2, 5)
>>> a[True].shape
(1, 2, 5)
>>> a[False].shape
(0, 2, 5)
```

This is different from what `a[np.nonzero(True)]` would
return:[^nonzero-scalar-footnote]

[^nonzero-scalar-footnote]: But note that this also wouldn't work if
    `np.nonzero(True)` returned the empty tuple `()`. In fact, there's no
    generic index that `np.nonzero()` could return that would be equivalent
    to the actual indexing behavior of a boolean scalar, especially for
    `False`.

<!-- TODO: Update this when NumPy 2.0 is released. -->
```py
>>> a[np.nonzero(True)].shape # doctest: +SKIP
(1, 5)
>>> a[np.nonzero(False)].shape # doctest: +SKIP
(0, 5)
```

The scalar boolean behavior may seem like an odd corner case. You might wonder
why NumPy supports using a `True` or `False` as an index, especially since it
has slightly different semantics than higher dimensional boolean arrays.

The reason scalar booleans are supported is that they are a natural
generalization of n-D boolean array indices. While the `np.nonzero()` rule
does not hold for them, the more general rule about replacing
`idx.ndim` dimensions a single dimension does.

Consider the most common case of using a boolean index: masking some subset of
the entire array. This typically looks something like
`a[some_boolean_expression_on_a] = mask_value`. For example:

```py
>>> a = np.asarray([[0, 1], [1, 0]])
>>> a[a == 0] = -1
>>> a
array([[-1,  1],
       [ 1, -1]])
```

Here, we set all the `0` elements of `a` to `-1`. We do this by creating the
boolean mask `a == 0`, which is a boolean expression created from `a`. Our
mask might be a lot more complicated in general, but it still is usually the
case that our mask is constructed from `a`, and thus has the exact same shape
as `a`. Therefore, `a[mask]` is a 1 dimensional array with
`np.count_nonzero(mask)` elements. In this example, this doesn't actually
matter because we are using the mask as the left-hand side of an assignment.
As long as the right-hand side is broadcast compatible with `a[mask]`, it will
be fine. In this case, it works because `-1` is a scalar, which is always
broadcast compatible with everything, but more generally we could index the
right-hand side with the exact same mask index to ensure it is exactly the
same shape as the left-hand side.

In particular, note that `a[a == 0] = -1` works no matter what the shape or
dimensionality of `a` is, and no matter how many `0` entries it has. Above
it had 2 dimensions and two `0`s, but it would also work if it were
1-dimensional:

```py
>>> a = np.asarray([0, 1, 0, 1])
>>> a[a == 0] = -1
>>> a
array([-1,  1, -1,  1])
```

Or if it had no actual `0`s:[^0-d-mask-footnote]

[^0-d-mask-footnote]: In this example, `a == 0` is `array([False, False,
    False])`, and `a[a == 0]` is an empty array of shape `(0,)`. The reason
    this works is that the right-hand side of the assignment is a scalar,
    i.e., NumPy casts it to an array of shape `()`. The shape `()` broadcasts
    with the shape `(0,)` to the shape `(0,)`, and so this is what gets
    assigned, i.e., "nothing" (of shape `(0,)`) gets assigned to "nothing" (of
    matching shape `(0,)`). This is one reason why [broadcasting
    rules](broadcasting) apply even to dimensions of size `0`.

```py
>>> a = np.asarray([1, 1, 2])
>>> a[a == 0] = -1
>>> a
array([1, 1, 2])
```

But even if `a` is a 0-D array, i.e., a single scalar value, we would still
expect this sort of thing to still work, since, as we said, `a[a == 0] = -1`
should work for *any* array. And indeed, it does:

```py
>>> a = np.asarray(0)
>>> a.shape
()
>>> a[a == 0] = -1
>>> a
array(-1)
```

Consider what happened here. `a == 0` is the a 0-D array `array(True)`.
`a[True]` is a 1-D array containing the single True value corresponding to
the mask, i.e., `array([0])`.

```py
>>> a = np.asarray(0)
>>> a[a == 0]
array([0])
```

This then gets assigned the value `-1`, which as a scalar, gets broadcasted
to the entire array, thereby replacing this single `0` value with `-1`. The
`0` in the masked array corresponds to the same `0` in memory as `a`, so the
assignment mutates it to `-1`.

If our 0-D `a` was not `0`, then `a == 0` would be `array(False)`. Then `a[a ==
0]` would be a 1-D array containing no values, i.e., a shape `(0,)` array:

```py
>>> a = np.asarray(1)
>>> a[a == 0]
array([], dtype=int64)
>>> a[a == 0].shape
(0,)
```

In this case, `a[a == 0] = -1` would assign `-1` to all the values in `a[a
== 0]`, i.e., no values, so `a` would remain unchanged:

```py
>>> a[a == 0] = -1
>>> a
array(1)
```

The point is that the underlying logic works out so that `a[a == 0] = -1`
always does what you'd expect: every `0` value in `a` is replaced with `-1`
*regardless* of the shape of `a`, including if that shape is `()`.

```{rubric} Footnotes
```
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
