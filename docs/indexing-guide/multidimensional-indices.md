# Multidimensional Indices

Unlike [slices](slices-docs) and [integers](integer-indices), which work not
only on NumPy arrays but also on built-in Python sequence types such as
`list`, `tuple`, and `str`, the remaining index types do not work at all on
non-NumPy arrays. If you try to use one on a `list`, for example, you will get
an `IndexError`. The semantics of these indices are defined by the NumPy
library, not the Python language.

(what-is-an-array)=
## What is an array?

Before we look at indices, let's take a step back and look at the NumPy array.
Just what is it that makes NumPy arrays so ubiquitous and makes NumPy one of
the most successful numerical tools ever? The answer is quite a few things,
which come together to make NumPy a fast and easy to use library for array
computations. But one in particular is multidimensional indexing.

Let's consider pure Python for a second. Suppose we have a list of values.
Say, these values correspond to your bowling scores.

```py
>>> scores = [70, 65, 71, 80, 73]
```

From [what we learned before](slices-docs), we can now index this list with
integers or slices to get some subsets of it.

```py
>>> scores[0] # The first score
70
>>> scores[-3:] # The last three scores
[71, 80, 73]
```

One can imagine all different sorts of things you'd want to do with your
scores that might involve selecting individual scores or ranges of scores (for
example, with the above examples, we could easily compute the average score of
our last three games, and see how it compares to our first game). So hopefully
you are convinced that at least the types of indices we have learned so far
are useful.

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

That's a mess. Clearly, you should have inverted the way you constructed your
list of lists, so that each inner list corresponds to a game, and each element
of that list corresponds to the person (for now, just you and Bob):

```py
>>> scores = [[70, 100], [65, 93], [71, 111], [80, 104], [73, 113]]
```

Now you can much more easily get the scores for the first game

```py
>>> scores[0]
[70, 100]
```

Except now if you want to look at just your scores for all games (that was
your original purpose after all, before Bob got involved), it's the same
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

In NumPy, these different levels of nesting are called *axes* or *dimensions*.
The number of axes---the level of nesting---is called the
*rank*[^rank-footnote] or *dimensionality* of the array. Together, the lengths
of these lists at each level are called the *shape* of the array (remember
that the lists at each level have to have the same number of elements).

[^rank-footnote]: Not to be confused with [mathematical definitions of
    rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)).

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

This is the power of multidimensional indexing in NumPy arrays. If we have a list of
lists of numbers, or a list of lists of lists of numbers, and so on, we can
index things at any "nesting level" equally easily. There is a small,
reasonable restriction, namely that each "level" of lists (axis)
must have the same number of elements. This restriction is reasonable because
in the real world, data tends to be tabular, like bowling scores, meaning each
axis will naturally have the same number of elements (and even if this weren't
the case, for instance, if Bob was out sick for a game, we could easily use a
sentinel value like `-1` or `nan` for a missing value).

The indexing semantics are only a small part of what makes NumPy arrays so
powerful. They have many other advantages as well, which are unrelated to
indexing. They operate on contiguous memory using native machine datatypes,
which makes them very fast. They can be manipulated using array expressions
with [broadcasting semantics](broadcasting); for example, you can easily add a
handicap to the scores array with something like `scores + np.array([124,
95])`, which would itself be a nightmare using the list of lists
representation. This, along with the powerful ecosystem of libraries like
`scipy`, `matplotlib`, and thousands of others, are what have made NumPy such
a popular and essential tool.

(basic-indices)=
## Basic Multidimensional Indices

First let's look at the basic multidimensional indices. These are the index
types that, unlike [integers](integer-indices) and [slices](slices), only make
sense for multidimensional arrays (and in particular, they will not work on
built-in Python types like lists and strings). There are three types of basic
multidimensional indices, [tuples](tuple-indices),
[ellipses](ellipsis-indices), and [newaxis](newaxis-indices) (or `None`).

(tuple-indices)=
### Tuples

The basic building block of multidimensional indexing is the `tuple` index. A
tuple index doesn't select elements on its own. Rather, it contains other indices,
which themselves select elements. The general rule for tuples is that each
element of a tuple index selects the corresponding elements for the
corresponding axis of the array (this rule is modified a little bit in the
presence of ellipses or newaxis, as we will see below).

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
The [slices](slices-docs) page stressed the point that [slices always keep the
axis they index](subarray), but it wasn't clear why that is important until
now. Suppose we slice the first axis of `a`, then later, we take that array
and want to get the first element of the last row.


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
because it means that you can index the array
uniformly.[^size-1-dimension-footnote] And this doens't apply just to
indexing. Many NumPy functions reduce the number of dimensions of the output
(for example, {external+numpy:func}`numpy.sum`), but they have a `keepdims`
argument to leave the result as dimension 1 instead.

[^size-1-dimension-footnote]: In this example, if we knew that we were always
    going to select exactly one element (say, the second one) from the first
    dimension, we could equivalently use `a[1, np.newaxis]` (see
    [](integer-indices) and [](newaxis-indices)). The advantage of this is
    that we would get an error if the first dimension of `a` doesn't actually
    have `2` elements, whereas `a[1:2]` would just silently give an [size 0 array](size-0-arrays).

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
    makes a distinction between scalars and 0-D (i.e., shape `()`) arrays. On either, an
    empty tuple index `()` will always produce a scalar, and a single ellipsis
    `...` will always produce a 0-D array:

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

    The difference between scalars and 0-D arrays in NumPy is subtle. In most
    contexts, they will both work identically, but there are some places where
    you need one and not the other, and the above trick can be used to convert
    between them. See footnotes [^integer-scalar-footnote] and [Other Topics
    Relevant to Indexing footnote 1](view-scalar-footnote-ref) for two
    important differences between the two which are related to indexing.

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
instead use `a[:, :, :, :, 0]`. This is not only tedious to type, but it makes
it impossible to write our index in a way that works for any number of
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
least* 3 dimensions (assuming of course that the first dimension is at least
`2` and the last dimension is at least `3`).

Above, we saw that a tuple index implicitly ends in some number of trivial `:`
slices. We can also see here that a tuple index also always implicitly ends in
an ellipsis, which serves the exact same purpose. Namely, **an ellipsis
automatically serves as a stand-in for the "correct" number of trivial `:`
slices to select the intermediate axes of an array**. And just as with the
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

The rules for an ellipsis are

- **An ellipsis index is written with three dots: `...`.**

- **`...` automatically selects 0 or more intermediate axes in an array.**

- **Every index before `...` operates on the first axes of the array. Every
  index after `...` operates on the last axes of the array.**

- **Every tuple index that does not have an ellipsis in it implicitly ends in
  `...`.**

- **At most `...` is allowed in a tuple index.**

(newaxis-indices)=
### newaxis

The final basic multidimensional index type is `newaxis`. `np.newaxis` is an
alias for `None`. Both `np.newaxis` and `None` work exactly the same, however,
`np.newaxis` is often more explicit than `None`, which may look odd in an
index, so it's generally preferred. However, you will often see `None` used
directly instead of `np.newaxis`, so it's important to remember that they are
the same thing.

```py
>>> print(np.newaxis)
None
>>> np.newaxis is None # They are exactly the same thing
True
```

`newaxis`, as the name suggests, adds a new axis. This new axis has size `1`.
The new axis is added in the corresponding location in the array. A size `1`
axis doesn't add or remove any of the elements from the array. Using the
[nested lists analogy](what-is-an-array), it just adds a new "layer" to the
list of lists.

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

When you include `newaxis` alongside other indices in a [tuple
index](tuple-indices), it doesn't affect which axes those indices select.
You can think of the `newaxis` index as inserting the new axis in-place in the
index, so that the other indices still select the same corresponding axes they
would select if it weren't there.

Take our example array, which has shape `(3, 2, 4)`

```py
>>> a = np.arange(24).reshape((3, 2, 4))
>>> a.shape
(3, 2, 4)
```

The index `a[0, :2]` has shape `(2, 4)`, because the first integer index `0`
removes the first axis, the slice index `:2` selects 2 elements from the
second axis, and the third axis is not selected at all, so it remains intact
with 4 elements.

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

In each case, the exact same elements are selected, that is, the `0` always
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

The only difference is where the shape 1 axis is inserted.

- `a[np.newaxis, 0, :2]`: the new axis is inserted before the first axis, but
the `0` and `:2` still index the original first and second axes. The resulting
shape is `(1, 2, 4)`.

- `a[0, np.newaxis, :2]`: the new axis is inserted after the first axis, but
because the `0` removes this axis when it indexes it, the resulting shape is
still `(1, 2, 4)` (and the resulting array is the same).

- `a[0, :2, np.newaxis]`: the new axis is inserted after the second axis,
because the `newaxis` comes right after the `:2`, which indexes the second
axis. The resulting shape is `(2, 1, 4)`. Remember that the `4` in the shape
corresponds to the last axis, which isn't represented in the index at all.
That's why in this example, the `4` still comes at the end of the resulting
shape.

- `a[0, :2, ..., np.newaxis]`: the `newaxis` is after an ellipsis, so the new
axis is inserted at the end of the shape. The resulting shape is `(2, 4, 1)`.

In general, in a tuple index, the axis that each index indices corresponds to
its position in the tuple index, after removing any `newaxis` indices
(equivalently, `newaxis` indices can be though of as adding new axes *after*
the existing axes are indexed).

A shape 1 axis can always be inserted anywhere in an array's shape without
changing the underlying elements.

An array index can include multiple `newaxis`'s, (or `None`'s). Each will add a
shape 1 axis in the corresponding location. Can you figure out what the shape
of

```py
a[np.newaxis, 0, newaxis, :2, newaxis, ..., newaxis]
```

will be (remember that `a.shape` is `(3, 2, 4)`)?

````{dropdown} Click here to show the solution

```py
>>> a[np.newaxis, 0, np.newaxis, :2, np.newaxis, ..., np.newaxis].shape
(1, 1, 2, 1, 4, 1)
```

````

To summarize,

> **`newaxis` (or `None`) inserts a new size 1 axis in the
corresponding location in the tuple index. The remaining, non-`newaxis`
indices in the tuple index are indexed as if the `newaxis` indices were not
there.**

#### Where `newaxis` is Used

What we haven't said yet is why you would want to do such a thing in the first
place. One use-case is to explicitly convert a 1-D vector into a 2-D matrix
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
of broadcasting is that size 1 dimensions are not directly useful, in the sense that
they could be removed without actually changing anything about the underlying
data in the array. So they are used as a signal that that dimension can be
repeated in operations. `newaxis` is therefore useful for inserting these size
1 dimensions in situations where you want to force your data to be repeated. For
example, suppose we have the two arrays

```py
>>> x = np.array([1, 2, 3])
>>> y = np.array([100, 200])
```

and suppose we want to compute an "outer" sum of `x` and `y`, that is, we want
to compute every combination of `i + j` where `i` is from `x` and `j` is from
`y`. The key realization here is that what we want is simply to
repeat each entry of `x` 3 times, to correspond to each entry of `y`, and
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
This can easily be done with `np.newaxis`.

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

Remember, as we saw [before](single-axis-tuple), size 1 dimensions may seem
redundant, but they are not a bad thing. Not only do they allow indexing an
array uniformly, they are also very important in the way they interact with
NumPy's broadcasting rules.


(advanced-indices)=
## Advanced Indices

Finally we come to the so-called advanced indices. These are "advanced" in the
sense that they are more complex. They are also distinct from "basic" indices
in that they always return a copy (see [](views-vs-copies)). Advanced indices
allow selecting arbitrary parts of an array, in ways that are impossible with
the basic index types. Advanced indexing is also sometimes called "fancy
indexing" or indexing by arrays, as the indices themselves in advanced
indexing are arrays:  [integer arrays](integer-array-indices) and [boolean
arrays](boolean-array-indices).

Using an array that does not have an integer
or boolean dtype as an index is an error.

```{note}
In this section, do not confuse the *array being indexed* with the *array that
is the index*. The former can be anything and have any dtype. It is only the
latter that is restricted to being integer or boolean.
```

(integer-array-indices)=
### Integer Arrays

Integer array indices are very powerful. Using them, it is possible to
construct effectively arbitrary new arrays consisting of elements from the
original indexed array.

Let's consider, as a start, a simple one-dimensional array:

```py
>>> a = np.array([100, 101, 102, 103])
```

Let's suppose we wish to construct from this array, the 2-D array

```
[[ 100, 102, 100 ],
 [ 103, 100, 102 ]]
```

using only indexing operations.

It should hopefully be clear that there's no way we could possibly construct
this array as `a[idx]`, using only the index types we've seen so far. For one
thing, [integer indices](integer-indices), [slices](slices-docs),
[ellipses](ellipsis-indices), and [newaxes](newaxis-indices) all only select
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

This is the array we wanted. We sort of constructed it using only indexing
operations, but we didn't actually do `a[idx]` for some index `idx`. Instead,
we just listed the index of each individual element.

An integer array index is basically this "cheating" method, except as a single
index. Instead of listing out `a[0]`, `a[2]`, and so on, we just create a
single integer array with those [integer indices](integer-indices):

```py
>>> idx = np.array([[0, 2, 0],
...                 [3, 0, 2]])
```

and then when we index `a` with this array, it works just as the index above:

```py
>>> a[idx]
array([[100, 102, 100],
       [103, 100, 102]])
```

This is how integer array indices work. **An integer array index can construct
*arbitrary* new arrays with elements from `a`, with the elements in any order
and even repeated, simply by indexing where each element of the new array
comes from.**

Note that `a[idx]` above is not the same size as `a` at all. `a` has 4
elements and is 1-dimensional, whereas `a[idx]` has 6 elements and is
2-dimensional. `a[idx]` also contains some duplicate elements from `a`, and
there are some elements which aren't selected at all. Indeed, we could take
*any* integer array of any shape, and as long as the elements are between 0
and 3, `a[idx]` would create a new array with the same shape as `idx` with
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
>>> a = np.array([[100, 101, 102],
...               [103, 104, 105]])
>>> idx = np.array([0, 0, 1])
>>> a[idx]
array([[100, 101, 102],
       [100, 101, 102],
       [103, 104, 105]])
>>> a[:, idx] # Index the second dimension
array([[100, 100, 101],
       [103, 103, 104]])
```

It would appear that this limits the ability to arbitrarily shuffle elements
of `a` using integer indexing. For instance, suppose we wanted to create the
array `[105, 100]` from the above `a`. Based on the above examples, it might
not seem possible. The elements `105` and `100` are not in the same row or
column of `a`. However, this is doable by providing multiple integer array
indices.

When multiple integer array indices are provided, the elements of each index
are correspondingly selected for that axis. It's perhaps most illustrative to
show this as an example. Given the above `a`, we can produce the array `[105,
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

A common use-case for integer array indexing is sampling. For example, to
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

Another common use-case of integer array indexing is to permute an array. An
array can be randomly permuted with
{external+numpy:meth}`numpy.random.Generator.permutation`. But what if we want
to permute two arrays with the same permutation? We can compute a permutation
index and apply it to both arrays. For a 1-D array `a` of size $n$, a
permutation index is just a permutation of the index `np.arange(k)`, which
itself is the [identity
permutation](https://en.wikipedia.org/wiki/identity_permutation) on `a`:

```py
>>> a = np.array([100, 101, 102, 103]) # as above
>>> b = np.array([200, 201, 202, 203]) # another array
>>> identity = np.arange(a.size)
>>> a[identity]
array([100, 101, 102, 103])
>>> rng = np.random.default_rng(11) # Seeded so this example reproduces
>>> random_permutation = rng.permutation(identity)
>>> a[random_permutation]
array([103, 101, 100, 102])
>>> b[random_permutation] # The same permutation on b
array([203, 201, 200, 202])
```

Now a few advanced notes about integer array indexing:

- Indices in the integer array can also be negative. Negative indices work the
  same as they do with [integer indices](integer-indices). Negative and
  nonnegative indices can be mixed arbitrarily.

  ```py
  >>> a = np.array([100, 101, 102, 103]) # as above
  >>> idx = np.array([0, 1, -1])
  >>> a[idx]
  array([100, 101, 103])
  ```

  If you want to convert an index with negative indices to an index without
  any negative indices, you can use the ndindex
  [`reduce()`](ndindex.IntegerArray.reduce) method with a shape.

- You can use a list instead of an array to represent an
  array.[^lists-footnote]. Using a list is useful if you are writing an array
  index by hand, but in all other cases, it is better to use an actual array
  instead. This will perform basic type checking (like that the shape and
  dtype are correct) when the index array is created rather than when the
  indexing happens. In most real-world use-cases, the index itself is
  constructed from some other array method.

  [^lists-footnote]: Beware that [versions of NumPy prior to
      1.23](https://numpy.org/doc/stable/release/1.23.0-notes.html#expired-deprecations)
      treated a single list as a [tuple index](tuple-indices) rather than as
      an array.


  ```py
  >>> a = np.array([100, 101, 102, 103]) # as above
  >>> a[[0, 1, -1]]
  array([100, 101, 103])
  >>> idx = np.array([0, 1, -1])
  >>> a[idx] # this is the same
  array([100, 101, 103])
  ```

- The integer arrays in an index need to either be the same shape or to be
  able to [broadcast](broadcasting) together to the same shape. The
  broadcasting behavior is useful if the index array would otherwise be
  repeated in a given dimension.

  It also means that if you mix an integer array index with a single
  [integer](integer-indices) index, it is the same as if you replaced the
  single integer index with an array of the same shape filled with that
  integer (because remember, a single integer index is the same thing as an
  integer array index of shape `()`).

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
  >>> # This is the same as
  >>> idx0_broadcasted = np.array([[1, 0], [1, 0], [1, 0]])
  >>> idx1_broadcasted = np.array([[0, 0], [1, 1], [2, 2]])
  >>> idx0_broadcasted.shape
  (3, 2)
  >>> idx1_broadcasted.shape
  (3, 2)
  >>> a[idx0_broadcasted, idx1_broadcasted]
  array([[103, 100],
         [104, 101],
         [105, 102]])
  ```

  And mixing an array and an integer index:

  ```
  >>> a
  array([[100, 101, 102],
         [103, 104, 105]])
  >>> idx0
  array([1, 0])
  >>> a[idx0, 2]
  array([105, 102])
  >>> # This is the same as
  >>> idx1 = np.array([2, 2])
  >>> a[idx0, idx1]
  array([105, 102])
  ```

  Here the `idx0` array specifies the indices along the first dimension, `1`
  and `0`, and the `2` species to always choose index `2` along the second
  dimension. This is the same as using the array `[2, 2]` for the second
  dimension, since this is the scalar `2` broadcasted to the shape of `[1,
  0]`.

  The ndindex method
  [`Tuple.broadcast_arrays()`](ndindex.Tuple.broadcast_arrays) (as well as
  [`expand()`](ndindex.Tuple.expand)) will broadcast array indices together
  into a canonical form.

[^integer-scalar-footnote]: In fact, if the integer array index itself has
    shape `()`, then the behavior is identical to simply using an `int` with
    the same value. So it's a true generalization. In ndindex,
    [`IntegerArray.reduce()`](ndindex.IntegerArray.reduce) will always convert
    a 0-D array index into an [`Integer`](ndindex.integer.Integer).

    However, there is one difference between `a[0]` and `a[asarray(0)]`. The
    latter is considered an advanced index, so it does not create a
    [view](views-vs-copies).

    ```py
    >>> a = np.empty((2, 3))
    >>> a[0].base is a
    True
    >>> print(a[np.array(0)].base)
    None
    ```

- As with all index types discussed in this guide, an integer array index can
  be used on the left-hand side of an assignment. This can be useful as it
  lets you surgically inject new elements into your array.

  ```py
  >>> a = np.array([100, 101, 102, 103]) # as above
  >>> idx = np.array([0, 3])
  >>> a[idx] = np.array([200, 203])
  >>> a
  array([200, 101, 102, 203])
  ```

  However, be careful, as there is inherent ambiguity here if your index array
  contains duplicate elements. For example, here we are saying to set index
  `0` to both `1` and `3`:

  ```py
  >>> a = np.array([100, 101, 102, 103]) # as above
  >>> idx = np.array([0, 1, 0])
  >>> a[idx] = np.array([1, 2, 3])
  >>> a
  array([  3,   2, 102, 103])
  ```

  The end result was `3`. This happened because `3` corresponded to the last
  `0` in the index array, but importantly, this is just an implementation
  detail. **NumPy makes no guarantees about the order in which index elements
  are assigned to.**[^cupy-assignment-footnote] If you are using an integer
  array as an assignment index, it's best to be careful to avoid duplicate
  entries in the index (or at least ensure that duplicate entries are always
  assigned the same value).

  [^cupy-assignment-footnote]: For example, in [CuPy](https://cupy.dev/),
    which implements the NumPy API on top of GPUs, [the behavior of this sort
    of thing is
    undefined](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__setitem__).
    This is because CuPy parallelizes the assignment operation on the GPU, and
    the element that gets assigned to the duplicate index "last" becomes
    dependent on a race condition.

- When one or more [slice](slices-docs), [ellipsis](ellipsis-indices), or
  [newaxis](newaxis-indices) indexes come before or after all the
  [integer](integer-indices) and integer array indices, the two sets of
  indices operate independently of one another. The slices and ellipses select
  the corresponding axes and newaxes add new axes to the corresponding
  locations, and the the integer array indices select the elements on their
  respective axes, as described above.

  For example, consider:

  ```py
  >>> a = np.array([[[100, 101, 102],  # Like above, but with an extra dimension
  ...                [103, 104, 105]]])
  ```

  This is the same `a` as in the above examples, except it has an extra size 1
  dimension:

  ```py
  >>> a.shape
  (1, 2, 3)
  ```

  We can select this first dimension with a slice `:`, then use the exact same
  index as in the example in the previous bullet:

  ```py
  >>> idx0
  array([1, 0])
  >>> a[:, idx0, 2]
  array([[105, 102]])
  >>> a[:, idx0, 2].shape
  (1, 2)
  ```

  The main benefit of this is that you can use `...` at the beginning of an
  index to select the last axes of an array with the integer array indices, or
  some number of `:`s to select some axes in the middle. This lets you do with
  indexing what you can also do with the {external+numpy:func}`numpy.take`
  function.

  To be sure, the slices can be any slice, and you can also include newaxes.
  This may potentially allow combining two sequential indexing operations into
  one, but they are mostly allowed for semantic completeness.

- Finally, if the [slice](slices-docs), [ellipsis](ellipsis-indices), or
  [newaxis](newaxis-indices) indices are *in between* the integer array
  indices, then something more strange happens. The two index types still
  operate "independently", but instead of the resulting array having the
  dimensions corresponding to the location of the indices, like in the
  previous bullet (and, indeed, as indexing works in every other instance),
  the shape corresponding to the (broadcasted) array indices (including
  integer indices) is *prepended* to the shape corresponding to the non-array
  indices. This is because there is inherent ambiguity in where these
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

  Here the (broadcasted) integer array index shape `(10, 20)` comes first in
  the result array and shape corresponding to the rest of the index, `(3, 4)`,
  comes last.

  If you find yourself running into this behavior, chances are you would be
  better off rewriting the indexing operation to be simpler. It's considered
  to be a design flaw of NumPy[^advanced-indexing-design-flaw-footnote], and
  it's not one that any other Python array library has copied. ndindex will
  raise a `NotImplementedError` exception on indices like these, because I
  don't want to deal with implementing this obscure logic.[^ndindex-advanced-indexing-design-flaw-footnote]

[^advanced-indexing-design-flaw-footnote]: Travis Oliphant, the original
    creator of NumPy, told me privately that "somebody should have slapped me
    with a wet fish" when he designed this.

[^ndindex-advanced-indexing-design-flaw-footnote]: I might accept a pull
    request implementing it, but I'm not going to do it myself.

#### Exercise

Given the above, you should be able to do the following exercise: how might
you randomly permute a 2-D array, using
{external+numpy:meth}`numpy.random.Generator.permutation` and indexing, in
such a way that each axis is permuted independently. This might correspond to
multiplying the array by random [permutation
matrices](https://en.wikipedia.org/wiki/Permutation_matrix) on the left and
right, like $P_1AP_2$. (Hint, one of the [basic indices](basic-indices)
discussed above may be useful here)

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

(note this isn't a full permutation of the array. For instance, the first row
`[5, 4, 7, 6]` contains only elements from the second row of `a`)

````{dropdown} Click here to show the solution

Suppose we have the following 2-D array `a`:

```py
>>> a = np.arange(12).reshape((3, 4))
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
```

We can generate permutations for the two axes using
{external+numpy:meth}`numpy.random.Generator.permutation` as
[above](permutation-example):

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
the same shape as `a` (`(3, 4)`). This should therefore be the (broadcasted)
shape of `idx0` and `idx1`, which are currently shapes `(3,)`, and `(4,)`. We
can use [`newaxis`](newaxis-indices) to insert dimensions so that they are
shape `(3, 1)` and `(1, 4)` so that they broadcast together to this shape.

```py
>>> a[idx0[:, None], idx1[None]]
array([[ 5,  4,  6,  7],
       [ 1,  0,  2,  3],
       [ 9,  8, 10, 11]])
```

You can check that this is a permutation of `a` where each axis is permuted
independently.

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
````

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
...     [False,  True, False],
...     [ True,  True, False],
...     [False, False, False]])
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
  {external+numpy:func}`np.count_nonzero(idx) <numpy.count_nonzero>`). `n` is
  always between `0` and `a.size`, inclusive.

- The selected elements are "in order". Namely, they are in C order. C order
  iterates the array `a` so that the last axis varies the fastest, like `(0,
  0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`, `(0, 1, 0)`, etc. This is also the order
  that the elements of `a` are printed in, and corresponds to the order they
  are selected in. Note, C ordering is always used, even when the underlying
  memory is not C ordered (see [](c-vs-fortran-ordering) for more details on C
  array order).

Usually these details are not important. That is because an array indexed by a
boolean array is usually only used indirectly, such as the left-hand side of
an assignment.

A typical use-case of boolean indexing is to create a boolean mask using the
array itself with some operators that return boolean arrays, such as relational
operators like as `==`, `>`, `!=`, and so on, logical operators like `&`
(and), `|` (or), `~` (not), and `^` (xor), and certain functions like `isnan`,
or `isinf`.

For example, take an example array of the integers from -10 to 10

```py
>>> a = np.arange(-10, 11)
>>> a
array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,
         3,   4,   5,   6,   7,   8,   9,  10])
```

Say we want to pick the elements of `a` that are both positive and odd. The
boolean mask `a > 0` represents which elements are positive and the boolean
mask `a % 2 == 1` represents which elements are odd. So our mask would be

```py
>>> mask = (a > 0) & (a % 2 == 1)
```

(Note the careful use of parentheses to match the [Python operator
precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence).
Masks must use the logical operators `&`, `|`, and `~` so that they can be
arrays. They cannot use the Python keywords `and`, `or`, and `not`, which only
operate on single Python booleans, not arrays.)

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
out). This can be easily done with an indexing assignment[^indexing-assignment-footnote]:

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

One common use-case of this is to mask out `nan` entries with a finite number,
like `0`:

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

It's important to not be fooled by this way of constructing a mask. Even
though the *expression* `(a > 0) & (a % 2 == 1)` depends on `a`, the resulting
array itself does not---it is just an array of booleans. **Boolean array
indexing, `a[mask]`, as with [all other types of indexing](what-is-an-index),
does not depend on the values of the array `a`, only in the positions of its
elements.**

This point might feel overly pedantic, but it matters once you realize that a
mask created with one array can be used on another array, so long as it has
the same shape. It's common to have multiple arrays representing different
data about the same set of points. You may wish to select a subset of one
array based on the value of the corresponding point in another array.

For example, suppose we wanted to plot some function, like $4x\sin(x) -
\frac{x^2}{4} - 2x$ on $[-10,10]$. We can set `x = np.linspace(-10, 10)` and
compute the array expression:

<!-- myst doesn't work with ```{plot}, and furthermore, if the two plot
directives are put in separate eval-rst blocks, the same plot is copied to
both. -->

```{eval-rst}
.. plot::
   :context: reset
   :include-source: True

   >>> import matplotlib.pyplot as plt
   >>> x = np.linspace(-10, 10, 10000) # 10000 evenly spaced points between -10 and 10
   >>> y = 4*x*np.sin(x) - x**2/4 - 2*x # some function
   >>> plt.scatter(x, y, marker=',', s=1)
   <matplotlib.collections.PathCollection object at ...>

If we want to show only those x values that are positive, we could easily do
this by modifying the ``linspace`` call that created ``x``. But what if we
want to show only those y values that are positive? The only way to do this is
to select them using a mask:

.. plot::
   :context: close-figs
   :include-source: True

   >>> plt.scatter(x[y > 0], y[y > 0], marker=',', s=1)
   <matplotlib.collections.PathCollection object at ...>

```

Here we are using the mask `y > 0` to select the corresponding values from
both the `x` *and* `y` arrays. Since the same mask is used on both arrays, the
values corresponding to this mask in both arrays will be selected. With `x[y >
0]`, even though the mask itself is not strictly created *from* `x`, it still
makes sense as a mask for the array `x`.  In this case, this mask selects a
nontrivial subset of `x`.

Using a boolean array mask created from a different array is very common. For
example, in [scikit-image](https://scikit-image.org/) an image is represented
as an array of pixel values. Masks can be used to select subset of the image,
and may be constructed based on the pixel values (e.g., all red pixels), or
based on a geometric shape independent of the pixel values (e.g., a
[circle](https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_camera_numpy.html)).
In machine learning, if `group` is an array with group numbers and `X` is an
array of features with repeated measurements per group, one can select the
features for a single group to do cross-validation like `X[group ==
0]`.

Now for the detailed semantics of boolean array indices:

- **A boolean array index will remove as many dimensions as the index has, and
  replace them with a single flat dimension, which has the size of the number
  of `True` elements in the index.** The shape of the boolean array index must
  match the dimensions that are being replaced.

  For example:

  ```py
  >>> a = np.arange(24).reshape((2, 3, 4))
  >>> idx = np.array([[True, False, True],
  ...                 [True, True, True]])
  >>> a.shape
  (2, 3, 4)
  >>> idx.shape
  (2, 3)
  >>> np.count_nonzero(idx)
  5
  >>> a[idx].shape
  (5, 4)
  ```

  This means that the final shape of an array indexed with a boolean mask
  depends on the value of the mask, specifically, the number of `True` values
  in it. It is easy to construct array expressions with boolean masks where
  the size of the array is impossible to know until runtime. For example:

  ```py
  >>> rng = np.random.default_rng(11) # Seeded so this example reproduces
  >>> a = rng.integers(0, 2, (3, 4))
  >>> a[a==0].shape # Could be any size from 0 to 12
  (7,)
  ```

  However, even if the number of elements of an indexed array is not knowable
  until runtime, the _number_ of dimensions is knowable. That's because a
  boolean mask acts as a flattening operation. The number of dimensions of the
  boolean array index are all removed from the indexed array and replaced with
  a single dimension (the *size* of this dimension, to be sure, cannot be
  known unless you know how many `True` elements there are in the index).

- **The order of the elements selected by a boolean array index `idx`
  corresponds to the elements being iterated in C-order** (see
  [](c-vs-fortran-ordering)). For example:

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

  In this example, the elements of `a` are ordered `0 1 2 ...` in C-order,
  which why in the final indexed array `a[idx]`, they are still in sorted
  order. C-order also corresponds to reading the elements of the array in the
  order that NumPy prints them in, from left to right (ignoring the brackets
  and commas).

- The actual order of a boolean mask is usually not that important. However,
  this fact has one important implication. **A boolean array index is the same
  as if you replaced `idx` with the result of
  {external+numpy:func}`np.nonzero(idx) <numpy.nonzero>` (unpacking the
  tuple)**, using the rules for [integer array indices](integer-array-indices)
  outlined above (although this rule *doesn't* apply to 0-dimensional
  boolean indices, see the next bullet point).


  ```py
  >>> np.nonzero(idx)
  (array([0, 0, 0, 1, 2, 2, 2]), array([0, 2, 3, 1, 0, 1, 3]))
  >>> idx0, idx1 = np.nonzero(idx)
  >>> a[idx0, idx1] # this is the same as a[idx]
  array([ 0,  2,  3,  5,  8,  9, 11])
  ```

  Here `np.nonzero(idx)` returns two integer array indices, one for each
  dimension of `idx`. These indices each have `7` elements, one for each
  `True` element of `idx`, and they select (in C-order), the corresponding
  elements. Another way to think of this is that `idx[np.nonzero(idx)]` will
  always return an array of (`np.count_nonzero(idx)`) `True`s, because
  `np.nonzero(idx)` is exactly the integer array indices that select the
  `True` elements of `idx`:

  ```py
  >>> idx[np.nonzero(idx)]
  array([ True,  True,  True,  True,  True,  True,  True])
  ```

  What this all means is that all the rules that are outlined above about
  [integer array indices](integer-array-indices), e.g., how they broadcast or
  combine together with slices, all also apply to boolean array indices after
  this transformation. This also specifies how boolean array indices and
  integer array indices combine together.

  Effectively, a boolean array index can be combined with other boolean array
  indices or integer or integer array indices by first converting the boolean
  index into integer indices (one for each dimension of the boolean index)
  that select each `True` element of the index, then broadcasting them all to
  a common shape.

  The particular thing to note here is that it is possible to use a boolean
  mask to select only a subset of the dimensions of `a`: TODO

  The ndindex method
  [`Tuple.broadcast_arrays()`](ndindex.Tuple.broadcast_arrays) (as well as
  [`expand()`](ndindex.Tuple.expand)) will convert boolean array indices into
  integer array indices via {external+numpy:func}`numpy.nonzero` and broadcast
  array indices together into a canonical form.


- A 0-dimensional boolean index (i.e., just the scalar `True` or `False`) is a
  little special. The `np.nonzero` rule stated above does not actually apply.
  That's because `np.nonzero` has odd behavior for 0-D arrays. `np.nonzero(a)` usually
  returns a tuple with as many arrays as dimensions of `a`:

  ```py
  >>> np.nonzero(np.array([True, False]))
  (array([0]),)
  >>> np.nonzero(np.array([[True, False]]))
  (array([0]), array([0]))
  ```

  But for a 0-D array, `np.nonzero(a)` doesn't return an empty tuple, but
  rather the same thing as `np.nonzero(np.array([a]))`:

  ```py
  >>> np.nonzero(np.array(False))
  (array([], dtype=int64),)
  >>> np.nonzero(np.array(True))
  (array([0]),)
  ```

  However, the key point, that a boolean array index removes and flattens
  `idx.ndim` dimensions from `a` is still True. Here, `idx.ndim` is `0`,
  because `array(True)` and `array(False)` have shape `()`. So what these
  indices do is remove 0 dimensions and add a single dimension of length 1 for
  True or 0 for False. Hence, if `a` has shape `(s1, ..., sn)`, then `a[True]`
  has shape `(1, s1, ..., sn)`, and `a[False]` has shape `(0, s1, ..., sn)`.

  ```py
  >>> a.shape # as above
  (3, 4)
  >>> a[True].shape
  (1, 3, 4)
  >>> a[False].shape
  (0, 3, 4)
  ```

  This breaks with what `a[np.nonzero(True)]` would give:[^nonzero-scalar-footnote]


  [^nonzero-scalar-footnote]: But note that this also wouldn't work if
      `np.nonzero(True)` returned the empty tuple `()`. In fact, there's no
      generic index that `np.nonzero()` could return that would be equivalent
      to the actual indexing behavior of a boolean scalar, especially for
      `False`.

  ```py
  >>> a[np.nonzero(True)]
  array([[0, 1, 2, 3]])
  ```

  This behavior may seem like an odd corner case, but it can come up in
  practice. TODO

  For one thing, it means that assigning to a masked index does the right thing
  if the mask expression happens to not actually contain the array.

  ```py
  TODO
  ```

## Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
