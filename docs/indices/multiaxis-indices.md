# Multiaxis Indices

Unlike [slices](slices-docs) and [integers](integer-indices), which work not
only on NumPy arrays but also on built-in Python sequence types such as
`list`, `tuple`, and `str`, the remaining index types do not work at all on
non-NumPy arrays. If you try to use one on a `list`, for example, you will get
an `IndexError`. The semantics of these indices are defined by the NumPy
library, not the Python language.

(what-is-an-array)=
## What is an array?

Before we look at indices, let's take a step back and look at the NumPy array.
Just what is it that makes NumPy arrays so ubiquitous and makes NumPy the most
successful numerical tools ever? The answer is quite a few things, which come
together to make NumPy a fast and easy to use library for array computations.
But one in particular is multidimensional indexing.

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
original purpose after all, before Bob got involved). And it's the same
problem again. To extract that you have to do

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

NumPy arrays work like a list of lists, but restricted so that these kinds of
things always "make sense". More specifically, if you have a "list of lists",
each element of the "outer list" must be a list. `[1, [2, 3]]` is not a valid
NumPy array. Furthermore, each inner list must have the same length, or more
precisely, the lists at each level of nesting must have the same length.

Lists of lists can be nested more than just two times. For example, you might
want to take your scores and create a new outer list, splitting them by
season. Then you would have a list of lists of lists, and your indexing
operations would look like `[[game[0] for game in season] for season in
scores]`.

In NumPy, these nested lists are called *axes*. The number of axes---the level
of nesting---is called the number of *dimensions*. Together, the lengths of
these lists at each level is called the *shape* of the array (remember that
each level has to have the same number of elements).

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

The shape of our array is a tuple with the number of games (the outer axis)
and the number of people (the inner axis).

```py
>>> scores.shape
(5, 2)
```

This is the power of multiaxis indexing in NumPy arrays. If we have a list of
lists of elements, or a list of lists of lists of elements, and so on, we can
index things at any "nesting level" equally easily. There is a small,
reasonable restriction, namely that each "level" (dimension) of lists (axis)
must have the same number of elements. This restriction is reasonable because
in the real world, data tends to be tabular, like bowling scores, meaning each
axis will naturally have the same number of elements (and even if this weren't
the case, for instance, if Bob was out sick for a game, we could easily use a
sentinel value like `-1` or `nan` for a missing value).

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

(tuple-indices)=
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
the first axis, the first element of the second axis, and the third element of
the third axis (remember that indexing is 0-based, so index `0` corresponds to
the first element, index `1` to the second, and so on). Looking at the list of
lists representation of `a` that was printed by NumPy:

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

Now, an important point about tuple indices should be made: **the parentheses
in a tuple index are completely optional.** Instead of writing `a[(1, 0, 2)]`,
we could have instead just wrote `a[1, 0, 2]`.

```py
>>> a[1, 0, 2]
10
```

These are exactly the same. When the parentheses are omitted, Python
automatically treats the index as a tuple. From here on out, we will always
omit the parentheses, as is common practice. Not only is this cleaner, it is
important to do so for another reason: syntactically, Python will not allow
slices in a tuple index if we include the parentheses:

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
are not directly inside of square brackets. If you really need to do this, you
can instead used the `slice` builtin function to create the equivalent tuple
`(slice(1), slice(None), slice(None, -1))`. But this is far less readable than
`1:, :, :-1`, so you should only do it if you are trying to generate an index
object separately from the array you are indexing (e.g., when using ndindex!).

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
with a single element, `a[i,]` is exactly the same index as that element,
`a[i]`.** The reason is that in both cases, the index `i` indexes over the
first axis of the array. This is true no matter what kind of index `i` is. `i`
can be an integer index, a slice, an ellipsis, and so on. With one exception,
that is: `i` cannot itself be a tuple index! Nested tuple indices are not
allowed.

In practice, this means that when working with NumPy arrays, you can think of
every index type as a single element tuple index. An integer index `0` is
*actually* the tuple index `(0,)`. The slice `a[0:3]` is actually a tuple
`a[0:3,]`. This is a good way to think about indices, because it will help you
to remember that non-tuple indices always operate as if they were the first
element of a single element tuple index, namely, the operate on the first axis
of the array (but also remember that this is not true for Python builtin
types. `l[0,]` and `l[0:3,]` will both error if `l` is a `list`, `tuple`, or
`str`).

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

`:` serves as a convenient way to "skip" axes. It is one of the most common
types of indices that you will see in practice for this reason. However, it is
important to remember that `:` is not special. It is just a slice, which picks
every element of the corresponding axis. We could also replace `:` with `0:n`,
where `n` is the size of the corresponding axis (see the [slices
documentation](omitted)).

```py
>>> a[0:3, 0:2, 0]
array([[ 0,  4],
       [ 8, 12],
       [16, 20]])
```

Of course, in practice using `:` is better because we might not know or care
what the actual size of the axis is, and it's less typing anyway.

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
`:` as there are remaining dimensions of the array.**[^tuple-slices-footnote]

[^tuple-slices-footnote]: A more mathematically precise way to say this might
    be this:  Suppose an array `a` has $n$ dimensions and a tuple index `i`
    has $k$ elements, where $k < n$. Then `a[i]` is exactly the same as
    `a[i2]`, where `i2` is `i` with $n - k$ trivial `:` slices appended to the
    end.

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

Thus, when it comes to indexing, all axes, even "trivial" axes, matter. It's
sometimes a good idea to maintain the same number of dimensions in an array
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

(ellipsis-indices)=
### Ellipses

Now that we understand how [tuple indices](tuple-indices) work, the remaining
basic index types are relatively straightforward. The first type of index we
will look at is the ellipsis. An ellipsis is written as literally three dots,
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

    This is also why the type name for the [ndindex `ellipsis`](ellipsis)
    object is lowercase, since `Ellipsis` is already a built-in name.

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
that it has 3 dimensions. If it instead of 5 dimensions, we would need to
instead use `a[:, :, :, :, 0]`. This is not only tedious, but it makes it
impossible to write our index in a way that works for any number of
dimensions. To contrast, if we wanted the first element of the *first* axis,
we could write `a[0]`, which works if `a` has 3 dimensions or 5 dimensions or
any number of dimensions.

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
the ellipsis will select the first axes of the array, and the indices after it
will select the last axes. The ellipsis automatically skips all intermediate
axes. For example, to select the first element of the first axis and the last
element of the last axis, we could use

```py
>>> a[0, ..., -1]
array([3, 7])
```

An ellipsis is also allowed to skip zero axes, if all the axes of the array
are already accounted for. For example, these are the same because `a` has 3
dimensions:

```py
>>> a[1, 0:2, 2]
array([10, 14])
>>> a[1, 0:2, ..., 2]
array([10, 14])
```

And indeed, the index `1, 0:2, ..., 2` will work with any array with *at
least* 3 dimensions (assuming the first dimension is at least `2` and the last
dimension is at least `3`).

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

Finally, only one ellipsis is allowed (otherwise it would be ambiguous which
axes are being indexed):

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

(newaxis-indices)=
### newaxis

The final basic index type is `newaxis`. `np.newaxis` is an alias for `None`.
Both `newaxis` and `None` work exactly the same, however, `newaxis` is often
more explicit than `None`, which may look odd in an index, so it's generally
preferred.

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
the index `a[0, :2]`.

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
but the `0` and `:2` still index the original first and second axes. The
resulting shape is `(1, 2, 4)`. In the second example, the new axis is
inserted after the first axis, but because the `0` removes this axis when it
indexes it, the resulting shape is still `(1, 2, 4)`. In the third example,
the new axis is inserted after the second axis, because the `newaxis` comes
right after the `:2`, which indexes the second axis. And in the fourth
example, the `newaxis` is after an ellipsis, so the new axis is inserted at
the end of the shape. In general, in a tuple index, the axis that an index
indices corresponds to its position in the tuple index, after removing any
`newaxis` indices (equivalently, `newaxis` indices can be though of as adding
new axes *after* the existing axes are indexed).

A shape 1 axis can always be inserted anywhere in an array's shape without
changing the underlying elements. This only corresponds to adding another
level of "nesting" to the array, when thinking of it as a list of lists.

An array index can include multiple `newaxis`'s, (or `None`'s). Each will add a
shape 1 axis in the corresponding location. Can you figure out what the shape
of `a[np.newaxis, 0, newaxis, :2, newaxis, ..., newaxis]` will be (remember
that `a.shape` is `(3, 2, 4)`)?[^newaxis-footnote]

[^newaxis-footnote]: Solution:

    ```py
    >>> a[np.newaxis, 0, np.newaxis, :2, np.newaxis, ..., np.newaxis].shape
    (1, 1, 2, 1, 4, 1)
    ```

A `newaxis` index by itself, `a[newaxis]`, simply adds a new axis at the
beginning of the shape.

To summarize, **`newaxis` (or `None`) inserts a new size 1 axis in the
corresponding location in the tuple index. The remaining, non-`newaxis`
indices in the tuple index are indexed as if the `newaxis` indices were not
there.**

What I haven't said yet is why you would want such a thing. One use-case is to
explicitly convert a 1-D vector into a 2-D matrix representing a row or
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

`v[newaxis]` inserts an axis at the beginning of the shape, making `v` a `(1,
3)` row vector. `v[..., newaxis]` inserts an axis at the end, making it a `(3,
1)` column vector.

But the most common usage is due to
[broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
Broadcasting is a powerful abstraction that applies to all operations in
NumPy. It allows arrays with mismatched shapes to be combined together as if
one or more of their dimensions were simply repeated the appropriate number of
times. Broadcasting is a generalization of this behavior

```py
>>> x = np.array([[1, 2], [3, 4]])
>>> x + 1
array([[2, 3],
       [4, 5]])
```

Here we can think of the scalar `1` as a shape `()` array, whereas `x` is a
shape `(2, 2)` array. Thus, `x` and `1` do not have the same shape, but `x +
1` is allowed via repeating `1` across every element of `x`. This means taking
`1` and treating it as if it were the shape `(2, 2)` array `[[1, 1], [1, 1]]`.

In general, broadcasting allows repeating only some dimensions. For example,
here we multiply `x`, a shape `(3, 2)` array, with `y`, a shape `(2,)` array.
`y` is virtually repeated into a shape `(3, 2)` array with each element of the
last dimension repeated 3 times.

```py
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
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

This is what the array `y` looks like before it is combined with `x` (except
the power of broadcasting is that the repeated entries are not literally
repeated in memory).

<!-- TODO: Write a separate page on broadcasting -->
(broadcasting)=
Broadcasting always happens automatically in NumPy whenever two arrays with
different shapes are combined, assuming those shapes are broadcast compatible.
The rule with broadcasting is that the shorter of the shapes are prepended
with length 1 dimensions so that they have the same number of dimensions. Then
any dimensions that are size 1 in a shape are replaced with the corresponding
size in the other shape. The other non-1 sizes must be equal or broadcasting
is not allowed. In the above example, we broadcast `(3, 2)` with `(2,)` by
first extending `(2,)` to `(1, 2)` then broadcasting the size `1` dimension to
the corresponding size in the other shape, `3`, giving a broadcasted shape of
`(3, 2)`. In more advanced examples, both shapes may have broadcasted
dimensions. For instance, `(3, 1)` can broadcast with `(2,)` giving `(3, 2)`.
The first shape would repeat the first axis 2 times along the second axis, and
the second would insert a new axis in the beginning that would repeat 3 times.
See the [NumPy
documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html) for
more examples of broadcasting.

The key idea of broadcasting is that size 1 axes are not directly useful, in
the sense that they could be removed without actually changing anything about
the underlying data in the array. So they are used as a signal that that
dimension can be repeated in operations. `newaxis` is therefore useful for
inserting these size 1 axes in situations where you want to force your data to
be repeated. For example, suppose we have the two arrays

```py
>>> x = np.array([1, 2, 3])
>>> y = np.array([100, 200])
```

and suppose we want to compute an "outer" sum of `x` and `y`, that is, we want
to compute every combination of `i + j` where `i` is from `x` and `j` is from
`y`. If we instead wanted to compute the outer product, we could use the
`np.outer` function, which does exactly this. But we instead want the sum. The
key realization here is that what we want is simply to repeat each entry of
`x` 3 times, to correspond to each entry of `y`, and respectively repeat each
entry of `y` 3 times to correspond to each entry of `x`. And this is exactly
the sort of thing broadcasting does! We only need to make the shapes of `x`
and `y` match in such a way that the broadcasting will do that. Since we want
both `x` and `y` to be repeated, we will need to broadcast both arrays. We
want to compute

```py
[[ x[0] + y[0], x[0] + y[1] ],
 [ x[1] + y[0], x[1] + y[1] ],
 [ x[2] + y[0], x[2] + y[1] ]]
```

That way the first dimension of the resulting array will correspond to values
from `x`, and the second dimension will correspond to values from `y`, i.e.,
`a[i, j]` will be `x[i] + y[j]`. Thus the resulting array will have shape `(3,
2)`. So to make `x` (shape `(3,)`) and `y` (shape `(2,)`) broadcast to this,
we need to make them `(3, 1)` and `(1, 2)`, respectively. This can easily be
done with `np.newaxis`.

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

Note: broadcasting automatically prepends shape `1` dimensions, so the
`y[np.newaxis, :]` operation is unnecessary.

```py
>>> x[:, np.newaxis] + y
array([[101, 201],
       [102, 202],
       [103, 203]])
```

Remember, as we saw [above](single-axis-tuple), size 1 axes may seem
redundant, but they are not a bad thing. Not only do they allow indexing an
array uniformly, they are also very important in the way they interact with
NumPy's broadcasting rules.

## Advanced Indices

Finally we come to the so-called advanced indices. These are "advanced" in the
sense that they are more complex. They allow selecting arbitrary parts of an
array, in ways that are impossible with the basic index types. Advanced
indexing is also sometimes called indexing by arrays, as there are two types
of advanced indices, both of which are arrays: integer arrays and boolean
arrays. Indexing by an array that does not have an integer or boolean dtype is
an error.

(integer-array-indices)=
### Integer Arrays

Integer array indices are very powerful. Using them, it is possible to
construct effectively arbitrary new arrays consisting of elements from the
original indexed array.

Let's consider, as a start, a simple one-dimensional array:

```py
>>> a = np.array([100, 101, 102, 103])
```

Let's suppose we wish to construct from this array, the array

```
[[ 100, 102, 100 ],
 [ 103, 100, 102 ]]
```

That is, a 2-D array with the elements in that given order.
This would be achieved by constructing an integer array index where the
corresponding elements of the index array are the [integer
index](integer-indices) of the elements in `a`. That is

```
>>> idx = np.array([[0, 2, 0], [3, 0, 2]])
>>> a[idx]
array([[100, 102, 100],
       [103, 100, 102]])
```

This is, how integer array indices work. You can shuffle the elements of `a`
into an arbitrary new array in arbitrary order simply by indexing where each
element of the new array comes from.

Note that `a[idx]` above is not the same size as `a` at all. `a` has 4
elements and is 1-dimensional, whereas `a[idx]` has 6 elements and is
2-dimensional. `a[idx]` also contains some duplicate elements from `a`, and
some elements which aren't selected at all. Effectively, we could take *any*
integer array of any shape, and as long as the elements are between 0 and 3,
`a[idx]` would create a new array with the same shape as `idx` with
corresponding elements selected from `a`.

A useful way to think about integer array indexing is that it generalizes
[integer indexing](integer-indices). With integer indexing, we are effectively
indexing using a 0-dimensional integer array, that is, a single
integer.[^integer-scalar-footnote] This always selects the corresponding
element from the given axis and removes the dimension. That is, it replaces that
dimension in the shape with `()`, the "shape" of the integer index.

Similarly, an integer array index always selects elements from the given axis,
and replaces the dimension in the shape with the shape of the array index. For
example:

```
>>> a = np.empty((3, 4))
>>> idx = np.zeros((2, 2), dtype=int)
>>> a[idx].shape
(2, 2, 4)
>>> a[:, idx].shape # Index the second dimension
(3, 2, 2)
```

When the indexed array `a` has more than one dimension, an integer array index
selects elements from a single axis.

```
>>> a = np.array([[100, 101, 102], [103, 104, 105]])
>>> a
array([[100, 101, 102],
       [103, 104, 105]])
>>> idx = np.array([0, 0, 1])
>>> a[idx]
array([[100, 101, 102],
       [100, 101, 102],
       [103, 104, 105]])
>>> a[:, idx] # Index the second dimension
array([[100, 100, 101],
       [103, 103, 104]])
```

It would appear now that this limits the ability to arbitrarily shuffle
elements of `a` using integer indexing. For instance, suppose we wanted to
create the array `[105, 100]` from the above `a`. Based on the above examples,
it might not seem possible. The elements 104 and 100 are not in the same row
or column of `a`. However, this is doable, by providing multiple
integer array indices.

When multiple integer array indices are provided, the elements of each index
are correspondingly selected for that axis. It's perhaps most illustrative to
show this as an example. Given the above `a`, we can produce the array `[104,
100]` using.

```
>>> idx = (np.array([1, 0]), np.array([2, 0]))
>>> a[idx]
array([105, 100])
```

Let's break this down. `idx` is a [tuple index](tuple-indices) with two
arrays, which are both the same shape. The first element of our desired
result, `105` corresponds to index `(1, 2)` in `a`:

```py
>>> a[1, 2]
105
```

So we write `1` in the first array and `2` in the second array. Similarly, the
next element, `100` corresponds to index `(0, 0)`, so we write `0` in the
first array and `0` in the second. In general, the first array contains the
indices for the first axis, the second array contains the indices for the
second axis, and so on. If we were to zip up our two index arrays, we would
get the set of indices for each corresponding element, `(1, 2)` and `(0, 0)`.

The resulting array has the same shape as our two index arrays. As before,
this shape can be arbitrary. Suppose we wanted to create the array

```
[[[ 102, 103],
  [ 102, 101]],
 [[ 100, 105],
  [ 102, 102]]]
```

Recall our `a`:

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

Now a few advanced notes about integer array indexing:

- Strictly speaking, the integer arrays only need to be able to broadcast
  together to the same shape.  This is useful if the index array would
  otherwise be repeated in a given dimension (see
  [broadcasting](broadcasting)).

  It also means that if you mix an integer array index with a single
  [integer](integer-indices) index, it is the same as if you replaced the
  single integer index with an array of the same shape filled with that
  integer (because remember, a single integer index is the same thing as an
  integer array index of shape `()`).

  For example:

[^integer-scalar-footnote]: In fact, if the integer array index itself has
    shape `()`, then the behavior is exactly identical to simply using an
    `int` with the same value. So it's a true generalization.
    In ndindex, [`IntegerArray.reduce()`](ndindex.IntegerArray.reduce)
    will always convert a 0-D array index into an
    [`Integer`](ndindex.integer.Integer).

(boolean-array-indices)=
### Boolean Arrays

The final index type is boolean arrays. Boolean array indices are also
sometimes called *masks*. We don't use that terminology here, to avoid
ambiguity with other types of array masking, but it's a useful way to think
about a boolean array index.

A boolean array index specifies which elements of an array should be selected
and which should not be selected.

The simplest and most common case is where a boolean array index has the same
shape as the array being indexed, and is the sole index (i.e., not part of a
larger [tuple index](tuple-indices)).

Consider the array:

```py
>>> a = np.arange(9).reshape((3, 3))
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
```

Suppose want to select the elements `1`, `3`, and `4`. To do so, we create a
boolean array of the same shape as `a` which is `True` in the positions where
those elements are and `False` everywhere else.

```py
>>> idx = np.array([
... [False,  True, False],
... [ True,  True, False],
... [False, False, False]])
>>> a[idx]
array([1, 3, 4])
```

From this we can see a few things:

- The result of indexing by the boolean mask is a 1-D array. If we think about
  it, this is the only possibility. A boolean index could select any number of
  elements. In this case, it selected 3 elements, but it could select as few
  as 0 and as many as 9 elements from `a`. So there would be no way to return
  a higher dimensional shape, or for the shape of the result to be related to
  the shape of `a`. The shape of `a[idx]` when `idx` is a boolean array is
  `(n,)` where `n` is the number of `True` elements in `idx` (i.e.,
  `np.count_nonzero(idx)`). `n` is always between `0` and `a.size`, inclusive.

- The selected elements are "in order". Namely, they are in C order. That is,
  C order iterates the array `a` so that the last axis varies the fastest, like
  `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`, `(0, 1, 0)`, etc. This is also the
  order that the elements of `a` are printed in, and corresponds to the order
  they are selected in. Note, C ordering is always used, even when the
  underlying memory is not C ordered (see [](c-vs-fortran-ordering) below).

Usually these details are not important. That is because an array indexed by a
boolean array is usually only used indirectly, such as the left-hand side of
an assignment.

A typical use-case of boolean indexing is to create a boolean mask using the
array itself with some operators that return boolean arrays, like relational
operators, such as `==`, `>`, `!=`, and so on, logical operators, such as `&`
(and), `|` (or), `~` (not), and `^` (xor), and certain functions like `isnan`,
or `isinf`.

For example, take an example array of the integers from -10 to 10

```py
>>> a = np.arange(-10, 11)
```

Say we want to pick the elements of `a` that are both positive and odd. The
boolean mask `a > 0` represents which elements are positive and the boolean
mask `a % 2 == 1` represents which elements are odd. So our mask would be

```py
>>> mask = (a > 0) & (a % 2 == 1)
```

(Note the careful use of parentheses. Masks must use the logical operators so
that they can be arrays. They cannot use the Python logical `and`, `or`, or
`not`.)

The `mask` is just an array of booleans:

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

Often one will see the `mask` written directly in the index, like

```py
>>> a[(a > 0) & (a % 2 == 1)]
array([1, 3, 5, 7, 9])
```

Suppose we wanted to set these elements of `a` to `-100` (i.e., to "mask" them
out). This can be easily done with an indexing assignment:

```
>>> a[(a > 0) & (a % 2 == 1)] = -100
>>> a
array([ -10,   -9,   -8,   -7,   -6,   -5,   -4,   -3,   -2,   -1,    0,
       -100,    2, -100,    4, -100,    6, -100,    8, -100,   10])
```

A common use-case of this is to mask out `nan` entries with a finite number,
like `0`.

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

Note that for this kind of use-case, the actual shape of `a[mask]` is
irrelevant. The important thing is that it is some subset of `a`, which is
then assigned to, mutating only those elements of `a`.

It's also important to not be fooled by this way of constructing a mask. Even
though the *expression* `(a > 0) & (a % 2 == 1)` depends on `a`, the resulting
array itself is just an array of booleans. **Boolean array indexing `a[mask]`,
as with [all other types of indexing](what-is-an-index), does not depend on
the values of the array `a`, only in the positions of its elements.**

This distinction matters when you realize that a mask created with one array
can be used on another array, so long as it has the same shape. For example,
suppose we wanted to plot `x + log(x - 1)` on [-5, 5]. We can set `x =
np.linspace(-5, 5)` and compute the array expression:

```
>>> x = np.linspace(-5, 5)
>>> y = x + np.log(x - 1)
```



## Other Topics Relevant to Indexing

(views-vs-copies)=
### Views vs. Copies

Advanced indices in NumPy also have a property that is important to make note
of in some situations, which is that they always create a **copy** of the
underlying array. This can matter for performance in some situations. It is
also relevant if you are modifying the array. For example:

```py
>>> a = np.arange(24).reshape((3, 2, 4))
>>> b = a[:, 0]
>>> c = a[np.array([[0, 0], [1, 1]])]
>>> c[:] = 0
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
<BLANKLINE>
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]],
<BLANKLINE>
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> b[:] = 0
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

Note that views are important for mutations in both directions. If `a` is a
view, mutating it will also mutate whichever array it is a view on, but
conversely, even if `a` is not a view, mutating it will modify any other
arrays which are views into `a`. It's best to minimize mutations in the
pretense of views, or to restrict them to a controlled part of the code, to
avoid unexpected "[action at a
distance](https://en.wikipedia.org/wiki/Action_at_a_distance_(computer_programming))"
bugs.

(c-vs-fortran-ordering)=
### C vs. Fortran ordering

NumPy has an internal distinction between C order and Fortran order.
C ordered arrays are stored in memory so that the last axis varies the
fastest. For example, if `a` has 3 dimensions, then its elements are stored in
memory like `a[0, 0, 0], a[0, 0, 1], a[0, 0, 2], ..., a[0, 1, 0], a[0, 1, 1], ...`. Fortran
ordering is the opposite: the elements are stored in memory so that the first axis varies
fastest, like `a[0, 0, 0], a[1, 0, 0], a[2, 0, 0], ..., a[0, 1, 0], a[1, 1, 0], ...`.[^c-order-footnote]

[^c-order-footnote]: C order and Fortran order are also sometimes row-major
  and column-major ordering, respectively. However, this terminology is
  confusing when the array has more than two axes or when it does not
  represent a mathematical matrix. It's better to think of them in terms of
  which axes vary the fastest---the last for C ordering and the first for
  Fortran ordering.

**The internal ordering of an array does not change any indexing semantics.**
The same index will select the same elements on `a` regardless of whether it
uses C or Fortran ordering internally.

Note that this also applies to [boolean array indices](boolean-array-indices),
even though they select elements in C order. A boolean mask always produces
the elements in C order, even if the underlying arrays use Fortran ordering.

```py
>>> a = np.arange(9).reshape((3, 3))
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> idx = np.array([
... [False,  True, False],
... [ True,  True, False],
... [False, False, False]])
>>> a[idx]
array([1, 3, 4])
>>> a_f = np.asarray(a, order='F')
>>> a_f # a_f looks the same as a, but the internal memory is ordered differently
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> idx_f = np.asarray(idx, order='F')
>>> # These are all the same as a[idx]
>>> a_f[idx]
array([1, 3, 4])
>>> a[idx_f]
array([1, 3, 4])
>>> a_f[idx_f]
array([1, 3, 4])
```

**What ordering does affect is the performance of certain operations.** In
particular, the ordering affects whether it is more optimal to index along the
first axis or last axis of an array. For example, `a[0]` selects the first
subarray along the first axis (recall that `a[0]` is a [view](views-vs-copies)
into `a`, so it references the exact same memory as `a`). For a C ordered
array, which is the default, this subarray is contiguous in memory. This is
because the indices on the last axes vary the fastest (i.e., are next to each
other in memory), so selecting a subarray of the first axis picks elements
which are still contiguous. Conversely, for a Fortran ordered array, `a[0]` is
not contiguous, but `a[..., 0]` is.

```
>>> a[0].data.contiguous
True
>>> a_f[0].data.contiguous
False
>>> a_f[..., 0].data.contiguous
True
```

Operating on memory that is contiguous allows the CPU to place the entire
memory in the cache at once, and as a result is more performant. This wont' be
visible for our example `a` above, which is small enough to fix in cache
entirely, but matters for larger arrays. Compare the time to sum along `a[0]`
or `a[..., 0]` for C and Fortran ordered arrays for a 3-dimensional array with
a million elements (using [IPython](https://ipython.org/)'s `%timeit`):

```py
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
Fortran ordering) is about 3 times faster.

NumPy indexing semantics tend to favor thinking about arrays using the C
order, as one does not need to use an ellipsis to select contiguous
subarrays. C ordering also matches the [list-of-lists
intuition](what-is-an-array) intuition of an array, since an array like
`[[0, 1], [2, 3]]` is stored in memory as literally `[0, 1, 2, 3]` with C
ordering.

C ordering is the default in NumPy when creating arrays with functions like
`asarray`, `ones`, `arange`, and so on. One typically only switches to
Fortran ordering when calling certain Fortran codes, or when creating an
array from another memory source that produces Fortran ordered data.

Regardless of which ordering you are using, it is worth structuring your data
so that operations are done on contiguous memory when possible.

## Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
