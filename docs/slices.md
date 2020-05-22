# Slices

Python's slice syntax is one of the more confusing parts of the language, even
to experienced developers. In this page, we should carefully break down the
rules for slicing, and examine just what it is that makes it so confusing.

There are two primary aspects of slices that make the confusing: confusing
conventions, and split definitions. By confusing conventions, we mean that
slice semantics have definitions that are often difficult to reason about
mathematically. These conventions were chosen for syntactic convenience, and
one can easily see for most of them how they lead to concise notation for very
common operations, but it remains nonetheless true that they can make figuring
out the *right* slice to use in the first place complicated. By branching
definitions, we mean that the definition of a slice takes on fundamentally
different meanings if the elements are negative, nonnegative, or `None`. This
again is done for syntactic convenience, but it means that as a user, you must
switch your mode of thinking about slices depending on the sign or type of the
arguments.

The ndindex library can help with much of this, especially for people
developing libraries that consume slices. But for end-users the challenge is
often just to write down a slice. Even if you rarely work with NumPy arrays,
you will most likely require slices to select parts of lists or strings as
part of the normal course of Python coding.

ndindex focuses on NumPy array index semantics, but everything on this page
equally applies to sliceable Python builtin objects like lists, tuples, and
strings. This is because on a single dimension, NumPy slice semantics are
identical to the Python slice semantics (NumPy only begins to differ from
Python for multi-dimensional indices).

## What is a slice?

In Python, a slice is a special syntax that is allowed only in an index, that
is, inside of square brackets proceeding an expression. A slice consists of
one or two colons, with either an expression or nothing on either side of each
colon. For example, the following are all valid slices on the object `a`:

```
a[x:y]
a[x:y:z]
a[:]
a[x::]
a[x::z]
```

Furthermore, for a slice `x:y:z` on Python or NumPy objects, there is an additional
semantic restriction, which is that the expressions `x`, `y`, and `z` must be
either integers or `None`. A term being `None` is syntactically equivalent to
it being omitted. For example, `x::` is equivalent to `x:None:None`. In the
discussions below we shall use "`None`" and "omitted" interchangeably.

It is worth mentioning that the `x:y:z` syntax is not valid outside of square
brackets, but slice objects can be created manually using the `slice` builtin.
You can also use the `ndindex.Slice` object if you want to perform more
advanced operations. The discussions below will just use `x:y:z` without the
square brackets for simplicity.

## Integer indices

To understand slices, it is good to first review how integer indices work.
Throughout this guide, we will use as an example this prototype list:

```math
a = [0, 1, 2, 3, 4, 5, 6]
```

`a` is the same as `range(7)` and has 7 elements.

The key thing to remember about indexing in Python, both for integer and slice
indexing, is that it is 0-based. This means that the indexes start at 0. This
is the case for all **nonnegative** indexes. For example, `a[3]` would pick
the **fourth** element of `a`, in this case, `3`.

```math
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}3{\phantom{,}}
    & \color{red}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{red}{6\phantom{,}}\\
\end{array}
```

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[3]
3
```

For **negative** integers, the indices index from the end of the array. These
indices are necessary 1-based (or rather, -1-based), since `0` already refers
to the first element of the array. `-1` chooses the last element, `-2` the
second-to-last, and so on. For example, `a[-3]` picks the **third-to-last**
element of `a`, in this case, `4`:

```math
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{index}}
    & \color{red}{-7\phantom{,}}
    & \color{red}{-6\phantom{,}}
    & \color{red}{-5\phantom{,}}
    & \color{red}{-4\phantom{,}}
    & \color{blue}{-3\phantom{,}}
    & \color{red}{-2\phantom{,}}
    & \color{red}{-1\phantom{,}}\\
\end{array}
```

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[-3]
4
```

An equivalent way to think about negative indices is that an index `a[-i]`
picks `a[len(a) - i]`, that is, you can subtract the negative index off of
the size of the array (for NumPy arrays, replace `len(a)` with the size of the
axis being sliced). For example, `len(a)` is `7`:

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> len(a)
7
>>> a[7 - 3]
4
```

Therefore, negative indexes are primarily a syntactic convenience that allows
one to specify parts of an array that would otherwise need to be specified in
terms of the size of the array.

## 0-based

## Wraparound (Negative Indexes)

## Half-open

## Clipping

## Steps

## Negative Steps

## Omitted Entries (`None`)
