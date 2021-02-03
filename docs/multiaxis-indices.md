# Multiaxis Indices

Unlike [slices](slices-docs) and [integers](integer-indices), which work not
only on NumPy arrays but also on built-in Python sequence types such as
`list`, `tuple`, and `str`, the remaining index types do not work at all on
non-NumPy arrays. If you try to use one on a `list`, for example, you will get
an `IndexError`. The semantics of these indices are defined by the NumPy
library, not by the Python language.

## What is an array?

Before we look at indices, let's take a step back and look at just what is a
NumPy array. Just what is it that makes NumPy arrays so ubiquitous and makes
NumPy the most successful numerical tools ever? The answer is quite a few
things, which come together to make NumPy a fast and easy to use library for
array computations. But one in particular is multidimensional indexing.

Let's consider pure Python for a second. Suppose we have a list of values.
Say, these values correspond to your bowling scores.

```py
>>> scores = [70, 65, 71, 80, 73]
```

From what we [learned before](slices-docs), we can now index this list with
integers or slices to get some subsets of it.

```py
>>> scores[0] # The first score
70
>>> scores[1:4] # second through fourth scores
[65, 71, 80]
```

Now suppose your bowling buddy Bob learns that you are keeping track of scores
and wants you to add his scores as well. He bowls with you, so his scores
correspond to the same games as yours. You could make a new list,
`bob_scores`, but this means storing a new variable. You've got a feeling you
are going to end up keeping track of a lot of people's scores. So instead, you
change your `scores` list from a list of scores to a list of lists. The first
inner list is your scores, and the second will be Bob's.

```py
>>> scores = [[70, 65, 71, 80, 73], [100, 93, 111, 104, 113]]
```

Now you can easily get your scores:

```py
>>> scores[0]
[70, 65, 71, 80, 73]
```

and Bob's scores:

```py
>>> scores[1]
[100, 93, 111, 104, 113]
```

But now there's a problem (aside from the obvious problem that Bob is a better
bowler than you). If you want to see what everyone's scores are for the first
game, you have to do something like this:

```py
>>> [p[0] for p in scores]
[70, 100]
```

That's a mess. Clearly, you should have inverted the list of lists, so that
each list corresponds to a game, and each element of that list corresponds to
the person (for now, just you and Bob):

```py
>>> scores = [[70, 100], [65, 93], [71, 111], [80, 104], [73, 113]]
```

Now you can much more easily get the scores for the first game

```py
>>> scores[0]
[70, 100]
```

Except now you want to look at just your scores for all games (that was your
original purpose after all, before Bob got you to add his data). And it's the
same problem again. To extract that you have to do

```py
>>> [game[0] for game in scores]
[70, 65, 71, 80, 73]
```

which is the same mess as above. What are you to do?

The NumPy array provides an elegant solution to this problem. Our idea of
storing the scores as a list of lists was a good one, but unfortunately, it
pushed the limits of what the Python `list` type was designed to do. Python
`list`s can store anything, be it numbers, strings, or even other lists.
If we want to tell Python to index a list that is inside of another list, we
have to do it manually, because the elements of the outer list might not even
be lists. For example, `l = [1, [2, 3]]` is a perfectly valid Python `list`, but
the expression `[i[0] for i in l]` is meaningless, because not every element
of `l` is a list.

NumPy arrays work like a list of lists, but restricted so that things always
"make sense". More specifically, if you have a "list of lists", each element
of the "outer list" must be a list. `[1, [2, 3]` is not a valid NumPy array.
Furthermore, each inner list must have the same length.

Lists of lists can be nested more than just two times. For example, you might
want to take your scores and create a new outer list, splitting them by
season. Then you would have a list of lists of lists, and your indexing
operations would look like `[[game[0] for game in season] for season in
scores]`.

In NumPy, these nested lists are called *axes*, and the number of axes is
called the number of *dimensions*. The lengths of each of these levels of
lists is called the *shape* of the array (remember that each level has to have
the same number of elements).

A NumPy array of our scores (using the last representation) looks like this

```py
>>> import numpy as np
>>> scores = np.array([[70, 100], [65, 93], [71, 111], [80, 104], [73, 113]])
```

Except for the `np.array()` call, it looks exactly the same as the list of
lists. But the difference is indexing. If we want the first game, as before,
we use `scores[0]`:

```py
>>> scores[0]
array([ 70, 100])
```

But if we want to find only our scores, instead of using a list comprehension,
we can simply use

```py
>>> scores[:, 0]
array([70, 65, 71, 80, 73])
```

The index contains two elements, the slice `:` and the integer index `0`. The
slice `:` says to take everything from the first axis (which represents
games), and the integer index `0` says to take the first element of the second
axis (which represents people).

The shape of our array is the number of games (the outer axis) and the number
of people (the inner axis).

```py
>>> scores.shape
(5, 2)
```

This is the power of multiaxis indexing in NumPy arrays. If we have a list of
lists of elements, or a list of lists of lists of elements, and so on, we can
index lists at any "nesting level" just as easily. There is a small,
reasonable restriction, namely that each "level" (dimension) of lists (axis)
must have the same number of elements. This restriction is reasonable because
in the real world, data tends to be tabular, like bowling scores, meaning each
axis will naturally have the same number of elements (and even if this isn't
the case, for instance, if Bob was out sick for a game, we could easily use a
sentinel value like `-1` for a missing value).

The indexing semantics are only a small part of what makes NumPy arrays so
powerful. They have many other advantages as well, which are unrelated to
indexing. They operate on contiguous memory using native machine datatypes,
which makes them very fast. They can be manipulated using array expressions
with broadcasting semantics. For example, you can easily add a handicap to the
scores array with something like `scores + np.array([124, 95])`, which would
itself be a nightmare using the list of lists representation. This, along with
the powerful ecosystem of libraries like `scipy`, `matplotlib`, and the
scikits, are really what have made NumPy such a popular and essential tool.

## Basic Multiaxis Indices

### Tuples

The basic building block of multiaxis indexing is the `tuple` index. A tuple
doesn't select elements on its own. Rather, it contains other indices, which
themselves select elements. The general rule for tuples is that each element
of a tuple index selects the corresponding elements for the corresponding axis
of the array (this rule is modified a little bit in the presence of ellipses
or newaxis, as we will see below).

For example, let's suppose we have the 3-dimensional array `a` with shape `(3,
2, 4)`. For simplicity, we'll define `a` as a reshaped `arange`, so that each
element is distinct and we can easily see which elements are selected.

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

We also see that integer indices remove the axis, and slices keep the axis
(even when the resulting axis has shape 1):

```py
>>> a[0].shape
(2, 4)
>>> a[2:].shape
(1, 2, 4)
```

A tuple index indices the corresponding element of the corresponding axis. So
for example, the index `(1, 0, 2)` grabs the element in the second element of
the first axis, the first element of the second axis, and the third element
of the third axis (remember that indexing is 0-based, so index `0` corresponds
to the first element, index `1` to the second, and so on). Looking up at the list of lists representation of `a` that
was printed by NumPy:

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
>>> a[(1, 0, 2)]
10
```

If we had stopped at an intermediate tuple, instead of getting an element, we
would have gotten the subarray that we accessed. For example, just `(1,)`
gives us the first intermediate array we looked at:

```py
>>> a[(1,)]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

And `(1, 0)` gives use the second intermediate array we looked at:

```py
>>> a[(1, 0)]
array([ 8,  9, 10, 11])
```

We can actually think of the final element, `10`, as being an array with shape
`()` (0 dimensions). Indeed, NumPy agrees with this idea:

```py
>>> a[(1, 0, 2)].shape
()
```

Now, an important point about tuple indices should be made. **The parentheses
in a tuple index are completely optional.** Instead of writing `a[(1, 0, 2)]`,
we could have instead just wrote `a[1, 0, 2]`.

```py
>>> a[1, 0, 2]
10
```

These are exactly the same. When the parentheses are omitted, Python
automatically treats the index as a tuple. From here on out, we will always
omit the parentheses. Not only is this cleaner, it is important to do so for
another reason: syntactically, Python will not allow slices in a tuple index
if we include the parentheses:

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

This is because the slice syntax using `:` is very special in Python. It is
only allowed directly inside of square brackets. When Python parses `a[(1:, :,
:-1)]`, it first looks at the inner `(1:, :, :-1)` and tries to parse that
separately. But this is not valid syntax, because the `:` slice expressions
are not directly inside of square brackets. To be sure, if you really need to
do this, you can instead used the `slice` builtin function to create the
equivalent tuple `(slice(1), slice(None, slice(None, -1))`. But this is far
less readable than `1:, :, :-1`, so you should only do it if you are trying to
generate an index object separately from the array you are indexing.


Now, let's go back and look at an example we just showed:

```py
>>> a[(1,)]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

You might have noticed something about this. It is picking the second element
of the first axis. But from what we said earlier, we can also do this just by
using the basic index `1`, which will operate on the first axis:

```py
>>> a[1] # Exactly the same thing as a[(1,)]
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

This illustrates the first important fact about tuple indices. **A tuple index
with a single element, `a[i,]` is exactly the same index as that element, `a[i]`.**
The reason is that in both cases, the index `i` indexes over the first axis of
the array. This is true no matter what kind of index `i` is. `i` can be an
integer index, a slice, an ellipsis, and so on. With one exception, that is.
`i` cannot itself be a tuple index. Nested tuple indices are not allowed. A
tuple index can contain any other index type, but not another tuple index
type.

In practice, this means that you can think of every index type as a single
element tuple index. An integer index `0` is *actually* the tuple index
`(0,)`. The slice `a[0:3]` is actually a tuple `a[0:3,]`. This is a good way
to think about indices, because it will help you to remember that non-tuple
indices always operate as if they were the first element of a single element
tuple index, namely, the operate on the first axis of the array (although
remember that this fact is not true for Python builtin types: `l[0,]` and
`l[0:3,]` will both error if `l` is a `list`, `tuple`, or `str`).

Up to now, we looked at the tuple index `(1, 0, 2)`, which selected a single
element. And we considered sub-tuples of this, `(1,)` and `(1, 0)`, which
selected subarrays. What if we want to select other subarrays? For example,
`a[1, 0]` selects the subarray with the second element of the first axis and
the first element of the second axis. What if instead we wanted the first
element of the *last* axis (axis 3).

We can do this with slices. In particular, the trivial slice `:` will select
every single element of an axis. So we want to select every element from the
first and second axis, and only the first element of the last axis, meaning
our index is `:, :, 0`:

```py
>>> a[:, :, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

`:` serves as a convenient way to "skip" axes. It is by far the most common
slice that you will see in indices for this reason. It is important to
remember that `:` is not special. It is just a slice, which picks every
element of the corresponding axis. We could also replace `:` with `0:n`, where
`n` is the size of the corresponding axis.

```py
>>> a[0:3, 0:2, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

Of course, in practice using `:` is better because the axis might not know or
care what the actual size of the axis is.

When we used the indices `(1,)` and `(1, 0)`, we saw that these indexed the
first and the first and second axes, respectively, and left the last axis/es
intact, producing subarrays. Another way of saying this is that the each tuple
index implicitly ended with `:` slices, one for each axis we didn't index:

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

This is a rule in general, **a tuple index implicitly ends in as many slices
`:` as there are remaining dimensions of the array.** Put another way, if an
array has $n$ dimensions and you use a tuple with $k$ elements where $k < n$,
then the index implicitly selects the entirety of the last $n - k$ axes, which
is exactly the same as if you had appended $n - k$ trivial `:` slices to the
end of the index.

(single-axis-tuple)=
The [slices](slices-docs) document stressed the point that slices always keep
the axis they index, but it wasn't clear why that is important until now.
Suppose we slice the first axis of `a`, then later, we take that array and
want to get the first element of the last row.


```py
>>> n = 2
>>> b = a[:n]
>>> b[-1, -1, 0]
12
```

Here `b = a[:2]` has shape `(2, 2, 4)`

```
>>> b.shape
(2, 2, 4)
```

But suppose instead we used a slice that only picked one element from the
first axis

```py
>>> n = 1
>>> b = a[:n]
>>> b[-1, -1, 0]
4
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

Thus, when it comes to indexing, all axes matter, even "trivial" axes. It's
often a good idea to maintain the same number of dimensions in an array
throughout a computation, even if one of them sometimes has size 1, simply
because it means that you can index the array uniformly.

There are two final facts about tuple indices that should be noted before we
move on to the other basic index types. First, as we noticed above, **if a
tuple index has more elements than there are dimensions in an array, it
produces an `IndexError`.**

Secondly, an array can be indexed by an empty tuple `()`. If we think about it
for a moment, we said that every tuple index implicitly ends in enough trivial
`:` slices to select the remaining axes of an array. That means that for an
array `a` with $n$ dimensions, an empty tuple index `a[()]` should be the same
as `a[:, :, … (n times)]`. This would select every element of every axis. In
other words, **the empty tuple index always just returns the entire array
unchanged.**[^tuple-ellipsis-footnote]

[^tuple-ellipsis-footnote]: There is one important distinction between the
    empty tuple index (`a[()]`) and a single ellipsis index (`a[...]`). NumPy
    makes a distinction between scalars and shape `()` arrays. On either, an
    empty tuple index `()` will always produce a scalar, and a single ellipsis
    `...` will always produce a shape `()` array:

    ```py
    >>> s = np.int64(0) # Scalar
    >>> x = np.array(0) # Shape () array
    >>> s[()]
    0
    >>> x[()]
    0
    >>> s[...]
    array(0)
    >>> x[...]
    array(0)
    ```

    This also applies for tuple indices that select a single element. If the
    tuple contains a (necessarily redundant) ellipsis, the result is a shape
    `()` array. Otherwise, the result is a scalar. With the example array:

    ```py
    >>> a[1, 0, 2]
    10
    >>> a[1, 0, 2, ...]
    array(10)
    ```

    The difference between scalars and shape `()` arrays in NumPy is subtle.
    In most contexts, they will both work identically, but there are some
    places where you need one and not the other, and the above trick can be
    used to convert between them.

### Ellipses

Now that we understand how tuple indices work, the remaining basic index types
are relatively straightforward. The first type of index we will look at is the
ellipsis. An ellipsis is written as literally three dots,
`...`.[^ellipsis-footnote]

[^ellipsis-footnote]: You can also write out the word `Ellipsis`, but this is
    discouraged. In older versions of Python, the three dots `...` were not
    valid syntax outside of the square brackets of an index, but as of Python
    3, `...` is valid anywhere, so it is unnecessary to use the spelled out
    `Ellipsis` an any context. The only reason I mention this is that if you
    type `...` at the interpreter, it will print "Ellipsis", and this explains
    why.

    ```py
    >>> ...
    Ellipsis
    ```

Let's go back to one of the examples above. To remind, we have our array `a`:

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

In one of the examples, we wanted to select only the first element of the last
axis. We saw that we could use the index `:, :, 0`:

```py
>>> a[:, :, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

However, this is only actually correct for our specific array, because we know
that it has 3 dimensions. If it instead of 5 dimensions, we would need to use
`:, :, :, :, 0`. This is not only tedious, but it makes it impossible to write
our index in a way that works for any number of dimensions. To contrast, if we
wanted the first element of the *first* axis, we could write `a[0]`, which
works if `a` has 3 dimensions or 5 dimensions or any $n \geq 1$ dimensions.

The ellipsis solves this problem. An ellipsis index skips all the axes of an
array to the end, so that the indices after it select the last axes of the
array.

```py
>>> a[..., 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

You are also allowed to put indices before the ellipsis. The indices before
the ellipsis will select the axes at the beginning of the array, and the
indices at the end will select the axes at the end. The ellipsis automatically
skips all intermediate axes. For example, to select the first element of the
first axis and the last element of the last axis, we could use

```py
>>> a[0, ..., -1]
array([3, 7])
```

An ellipsis is also allowed to select zero axes, if all the axes of the array
are already accounted for. For example:

```py
>>> a[1, 0:2, 2]
array([10, 14])
>>> a[1, 0:2, ..., 2]
array([10, 14])
```

Above, we saw that a tuple index implicitly ends in some number of trivial `:`
slices. We can also see here that a tuple index also always implicitly ends in
an ellipsis, which serves the exact same purpose. Namely, an ellipsis
automatically serves as a stand-in for the "correct" number of trivial `:`
slices, to select the intermediate axes of an array. And just as with the
empty tuple index `()`, which we saw is the same as writing the right number
of trivial `:` slices, a single ellipsis and nothing else is the same as
selecting every axis of the array, i.e., it leaves the array intact.[^tuple-ellipsis-footnote]

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

Finally, only one ellipsis is allowed (otherwise it would be ambiguous in
general which axis is being indexed):

```py
>>> a[0, ..., 1, ..., 2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: an index can only have a single ellipsis ('...')
```

The rules for an ellipsis are

- **An ellipsis index is written with three dots: `...`.**
- **An ellipsis index automatically selects 0 or more intermediate axes in an
  array.**
- **Every index before an ellipsis operates on the first axes of the array.
  Every index after an ellipsis operates on the last axes of the array.**
- **Every tuple index that does not have an ellipsis in it implicitly ends in
  an ellipsis index.**
- **At most one ellipsis index is allowed in a tuple index.**

### newaxis

The final basic index type is `newaxis`. `np.newaxis` is an alias for `None`.
Both work exactly the same. However, `newaxis` is often more explicit than
`None`, which may look odd in an index, so it's generally preferred.

`newaxis`, as the name suggests, adds a new axis. This new axis has size `1`.
The new axis is added in the corresponding location in the array. Take our
example array, which has shape `(3, 2, 4)`

```py
>>> a = np.arange(24).reshape((3, 2, 4))
>>> a.shape
(3, 2, 4)
```

The index `a[0, :2]` has shape `(2, 4)`, because the first integer index `0`
removes the first axis, and the slice index `:2` selects 2 elements from the
second axis.

```py
>>> a[0, :2]
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])
>>> a[0, :2].shape
(2, 4)
```

Now look at the shape of `a` when we insert `newaxis` in various locations in
the index.

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

In each case, the exact same elements are indexed, that is, the `0` always
indexes the first axis and the `:2` always indexes the second axis.

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

The only difference is where the shape 1 axis is inserted. In the first
example, `a[newaxis, 0, :2]`, the new axis is inserted before the first axis,
but the 0 and :2 still index the original first and second axes. The resulting
shape is `(1, 2, 4)`. In the second example, the newaxis is inserted after the
first axis, but because the `0` removes the first axis when it indexes it, the
resulting shape is still `(1, 2, 4)`. In general, in a tuple index, the axis
that an index indices corresponds to its position in the tuple index, after
removing any `newaxis` indices (equivalently, newaxis indices can be though of
as adding new axes *after* the existing axes are indexed).

A shape 1 axis can always be inserted anywhere in an array's shape without
changing the underlying elements. This only corresponds to adding another
level of "nesting" to the array, when thinking of it as a list of lists.

An array index can include multiple newaxes, (or `None`s). Each will add a
shape 1 axis in the corresponding location. Can you figure out what the shape
of `a[np.newaxis, 0, newaxis, :2, newaxis, ..., newaxis]` will be?[^newaxis-footnote]

[^newaxis-footnote]: Solution:

    ```py
    >>> a[np.newaxis, 0, np.newaxis, :2, np.newaxis, ..., np.newaxis]
    array([[[[[[0],
               [1],
               [2],
               [3]]],
    <BLANKLINE>
    <BLANKLINE>
             [[[4],
               [5],
               [6],
               [7]]]]]])
    >>> a[np.newaxis, 0, np.newaxis, :2, np.newaxis, ..., np.newaxis].shape
    (1, 1, 2, 1, 4, 1)
    ```

A newaxis index by itself, `a[newaxis]`, simply adds a newaxis at the
beginning of an array.

To summarize, **`newaxis` (or `None`) inserts a new, size 1 axis in the
corresponding location in the tuple index. The remaining, non-newaxis indices
in the tuple index are indexed as if the `newaxis` indices were not there.**

What I haven't said yet is why you would want such a thing. One use case is to
explicitly convert a dimension 1 vector into a 2d matrix representing a row or
column vector. For example,

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

`v[newaxis]` inserts an axis at the beginning, making `v` a `(1, 3)` row
vector. `v[..., newaxis]` inserts an axis at the end, making it a `(3, 1)`
column vector.

Another common usage is due to broadcasting. Suppose we have the arrays

```py
>>> x1 = np.array([[1, 2, -1], [0, 0, 1]])
>>> x2 = np.array([[[2, 3, 3]], [[5, 2, 0]]])
>>> x1.shape
(2, 3)
>>> x2.shape
(2, 1, 3)
```

Suppose we want to add each element of `x1` to the corresponding element of
`x2`, that is, `1 + 2`, `2 + 3`, `-1 + 3`, and so on. If we just add `x1 +
x2`, this gives us something else:

```py
>>> x1 + x2
array([[[ 3,  5,  2],
        [ 2,  3,  4]],
<BLANKLINE>
       [[ 6,  4, -1],
        [ 5,  2,  1]]])
>>> (x1 + x2).shape
(2, 2, 3)
```

What happened is that NumPy's [broadcasting
rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) converted
the shape `(2, 3)` and `(2, 1, 3)` arrays to shape `(2, 2, 3)`. What we really
want is for the shapes to match, so that it doesn't broadcast. We can do this
in two ways, by deleting the second axis of `x2` using `x2[:, 0]` (remember
that integer indices will remove an axis, so `0` effectively deletes a size 1
axis)

```py
>>> x2[:, 0].shape
(2, 3)
>>> x1 + x2[:, 0]
array([[3, 5, 2],
       [5, 2, 1]])
>>> (x1 + x2[:, 0]).shape
(2, 3)
```

or by inserting a new axis into `x1`:

```py
>>> x1[:, np.newaxis].shape
(2, 1, 3)
>>> x1[:, np.newaxis] + x2
array([[[3, 5, 2]],

       [[5, 2, 1]]])
>>> (x1[:, np.newaxis] + x2).shape
(2, 1, 3)
```

If we want our end result to look like `x1`, we should choose the former
option, but if we want it to look like `x2`, we should insert the intermediate
axis.

Remember, as we saw [above](single-axis-tuple), size 1 axes may seem
redundant, but they are not a bad thing. They allow indexing an array
uniformly, and they are very important in the way the interact with NumPy's
[broadcasting
rules](https://numpy.org/doc/stable/user/basics.broadcasting.html). And in
general, it's often important that an array has the correct number of
dimensions, and adding size 1 axes to an array with fewer dimensions is
sometimes the correct way to achieve this.

## Advanced Indices

## Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
