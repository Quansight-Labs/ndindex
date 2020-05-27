Slices
======

Python's slice syntax is one of the more confusing parts of the
language, even to experienced developers. In this page, we should
carefully break down the rules for slicing, and examine just what it is
that makes it so confusing.

There are two primary aspects of slices that make the confusing:
confusing conventions, and split definitions. By confusing conventions,
we mean that slice semantics have definitions that are often difficult
to reason about mathematically. These conventions were chosen for
syntactic convenience, and one can easily see for most of them how they
lead to concise notation for very common operations, but it remains
nonetheless true that they can make figuring out the *right* slice to
use in the first place complicated. By branching definitions, we mean
that the definition of a slice takes on fundamentally different meanings
if the elements are negative, nonnegative, or `None`. This again is done
for syntactic convenience, but it means that as a user, you must switch
your mode of thinking about slices depending on the sign or type of the
arguments. There is no uniform formula that applies to all slices.

The ndindex library can help with much of this, especially for people
developing libraries that consume slices. But for end-users the
challenge is often just to write down a slice. Even if you rarely work
with NumPy arrays, you will most likely require slices to select parts
of lists or strings as part of the normal course of Python coding.

ndindex focuses on NumPy array index semantics, but everything on this
page equally applies to sliceable Python builtin objects like lists,
tuples, and strings. This is because on a single dimension, NumPy slice
semantics are identical to the Python slice semantics (NumPy only begins
to differ from Python for multi-dimensional indices).

What is a slice?
----------------

In Python, a slice is a special syntax that is allowed only in an index,
that is, inside of square brackets proceeding an expression. A slice
consists of one or two colons, with either an expression or nothing on
either side of each colon. For example, the following are all valid
slices on the object `a`:

    a[x:y]
    a[x:y:z]
    a[:]
    a[x::]
    a[x::z]

Furthermore, for a slice `x:y:z` on Python or NumPy objects, there is an
additional semantic restriction, which is that the expressions `x`, `y`,
and `z` must be either integers or `None`. A term being `None` is
syntactically equivalent to it being omitted. For example, `x::` is
equivalent to `x:None:None`. In the discussions below we shall use
"`None`" and "omitted" interchangeably.

It is worth mentioning that the `x:y:z` syntax is not valid outside of
square brackets, but slice objects can be created manually using the
`slice` builtin. You can also use the `ndindex.Slice` object if you want
to perform more advanced operations. The discussions below will just use
`x:y:z` without the square brackets for simplicity.

(integer-indices)=

Integer indices
---------------

To understand slices, it is good to first review how integer indices
work. Throughout this guide, we will use as an example this prototype
list:

$$
a = [0, 1, 2, 3, 4, 5, 6].
$$

`a` is the same as `range(7)` and has 7 elements.

The key thing to remember about indexing in Python, both for integer and
slice indexing, is that it is 0-based. This means that the indexes start
at 0. This is the case for all **nonnegative** indexes. For example,
`a[3]` would pick the **fourth** element of `a`, in this case, `3`.

$$
\begin{aligned}
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
\end{aligned}
$$

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[3]
3
```

For **negative** integers, the indices index from the end of the array.
These indices are necessary 1-based (or rather, -1-based), since 0
already refers to the first element of the array. `-1` chooses the last
element, `-2` the second-to-last, and so on. For example, `a[-3]` picks
the **third-to-last** element of `a`, in this case, `4`:

$$
\begin{aligned}
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
\end{aligned}
$$

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[-3]
4
```

An equivalent way to think about negative indices is that an index
`a[-i]` picks `a[len(a) - i]`, that is, you can subtract the negative
index off of the size of the array (for NumPy arrays, replace `len(a)`
with the size of the axis being sliced). For example, `len(a)` is `7`, so
`a[-3]` is the same as `a[7 - 3]`:

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> len(a)
7
>>> a[7 - 3]
4
```

Therefore, negative indexes are primarily a syntactic convenience that
allows one to specify parts of an array that would otherwise need to be
specified in terms of the size of the array.

If an integer index is greater than or equal to the size of the array,
or less than negative the size of the array (`i < len(a)`
or `i >= len(a)`), then it is out of bounds and will raise
an `IndexError`.

```py
>>> a[7]
Traceback (most recent call last):
...
IndexError: list index out of range
>>> a[-8]
Traceback (most recent call last):
...
IndexError: list index out of range
```

Points of Confusion
-------------------

The full definition of a slice could be written down in a couple of
sentences, although the branching definitions would necessitate several
"if" conditions. The [NumPy
docs](https://numpy.org/doc/stable/reference/arrays.indexing.html) on
slices say

> The basic slice syntax is `i:j:k` where *i* is the starting index, *j*
> is the stopping index, and *k* is the step ( $k\neq 0$ ). This
> selects the `m` elements (in the corresponding dimension) with index
> values *i, i + k, ..., i + (m - 1) k* where $m = q + (r\neq0)$ and
> *q* and *r* are the quotient and remainder obtained by dividing *j -
> i* by *k*: *j - i = q k + r*, so that *i + (m - 1) k \< j*.

While notes like this may give a technically accurate description of
slices, they aren't especially helpful to someone who is trying to
construct a slice from a higher level of abstraction such as "I want to
select this particular subset of my array".

Instead, we shall examine slices by carefully going over all the various
aspects of the syntax and semantics that can lead to confusion, and
attempting to demystify them through simple rules.

### Subarray

A slice always chooses a sub-array (or sub-list, sub-tuple, sub-string, etc.).
What this means is that a slice will always *preserve* the dimension that is
sliced. This is true even if a slice chooses only a single element, or even if
it chooses no elements. This is also true for lists and tuples. This is
different from integer indices, which always remove the dimension that they
index.

For example

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[3]
3
>>> a[3:4]
[3]
>>> a[5:2] # Empty slice
[]
>>> import numpy as np
>>> arr = np.array([[1, 2], [3, 4]])
>>> arr[0].shape # Removes the first dimension
(2,)
>>> arr[0:1].shape # Preserves the first dimension
(1, 2)
```

One consequence of this is that, unlike integer indices, slices will
never raise `IndexError`. Therefore you cannot rely on
runtime errors to alert you to coding mistakes relating to slice bounds
that are too large. See also the section on
[clipping](#clipping) below.

(0-based)=
### 0-based

For the slice `a:b`, with `a` and
`b` nonnegative integers, the indexes `a` and
`b` are 0-based, just as with
[integer indexing](integer-indices)
(although one should be careful that even though `b` is
0-based, the end slice is not included in the slice. See
[below](half-open)).

$$
\begin{aligned}
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{3\phantom{,}}
    & \color{blue}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{red}{6\phantom{,}}\\
\end{array}
\end{aligned}
$$

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[3:5]
[3, 4]
```

(half-open)=
### Half-open

Slices behave like half-open intervals. What this means is that the `end` in
`start:end` is *never* included in the slice (the exception is if `end` is
`None` or omitted, which always slices to the beginning or end of the array,
see [below](omitted)).

For example, `a[3:5]` slices the elements 3 and 4, but not 5 ([0-based](0-based)).

$$
\require{enclose}
\begin{aligned}
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{\enclose{circle}{3}}
    & \color{blue}{\enclose{circle}{4}}
    & \color{red}{\enclose{circle}{5}}
    & \color{red}{6\phantom{,}}\\
\end{array}
\end{aligned}
$$

```py
>>> a = [0, 1, 2, 3, 4, 5, 6]
>>> a[3:5]
[3, 4]
```

The half-open nature of slices means that you must always remember that the
`end` slice element is not included in the slice. However, it has a few
advantages

- The maximum length of a slice `start:end`, when `start` and `end` are
  nonnegative, is always `end - start` (the caveat "maximum" is here because
  if `end` extends beyond the end of the array, then `start:end` will only
  slice up to `len(a) - start`, see [below](clipping)). For example, `a[i:i+n]`
  will slice `n` elements from the array `a`.
- `len(a)` can be used as an end value to slice to the end of the array. For
  example, `a[1:len(a)]` slices from the second element to the end of the
  array. This is equivalent to `a[1:]`.

  ```py
  >>> a[1:len(a)]
  [1, 2, 3, 4, 5, 6]
  >>> a[1:]
  [1, 2, 3, 4, 5, 6]
  ```

- Consecutive slices can be appended to one another by making each successive
  slice's start the same as the previous slice's end. For example, for our
  list `a`, `a[2:3] + a[3:5]` is the same as `a[2:5]`.

  ```py
  >>> a[2:3] + a[3:5]
  [2, 3, 4]
  >>> a[2:5]
  [2, 3, 4]
  ```

  A common usage of this is to split a slice into two slices. For example, the
  slice `a[i:j]` can be split as `a[i:k]` and `a[k:j]`.

#### Wrong Ways of Thinking about Half-open Semantics

A note with half-open semantics. **The proper rule to remember for slices is
"the end is not included".**

There are several alternative ways that one might think of slice semantics,
but they are all wrong in subtle ways. To be sure, for each of these, one
could "fix" the rule by adding some conditions, "it's this in the case where
such and such is nonnegative and that when such and such is negative, and so
on". But that's not the point. The goal here is to *understand* slices.
Remember that one of the reasons that slices are difficult to understand is
these branching rules. By trying to remember a rule that has branching
conditions, you open yourself up to confusion. The rule becomes much more
complicated than it appears at first glance, making it hard to remember. You
may forget the "uncommon" cases and get things wrong when they come up in
practice.

Rather, it is best to remember the simplest rule possible that is *always*
correct. That rule is, "the end is not included". That is always right,
regardless of what the values of `start`, `end`, or `step` are. The only
exception is if `end` is `None`/omitted. In this case, the rule obviously
doesn't apply as-is, and so you can fallback to the next rule about omitted
start/end (see [below](omitted)).

**Wrong Rule 1: "a slice `a[start:end]` slices the half-open interval
$[\text{start}, \text{end})$ (equivalently, a slice `a[start:end]` picks the
elements `i` such that `start <= i < end`).** This is *only* the case if the
step size is positive. It also isn't directly true for negative `start` or
`end`. For example, with a step of -1, `a[start:end:-1]` slices starting at
`start` going in reverse order to `end`, but not including `end`.
Mathematically, this creates a half open interval $(\text{end}, \text{start}]$
(except reversed).

For example, say way believed that `a[5:3:-1]` sliced the half-open interval
$[3, 5)$ but in reverse order.

$$
\begin{aligned}
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{3\phantom{,}}
    & \color{blue}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{red}{6\phantom{,}}\\
\color{red}{\text{WRONG}}&
    &
    &
    & [\phantom{3,}
    & \tiny{\text{(reversed)}}
    & )
    & \\
\end{array}
\end{aligned}
$$

We might assume we would get

```py
>> a[5:3:-1]
[4, 3] # WRONG
```

Actually, what we really get is

```py
>>> a[5:3:-1]
[5, 4]
```

This is because the slice `5:3:-1` starts at index `5` and steps backwards to
index `3`, but not including `3`.

$$
\require{enclose}
\begin{aligned}
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{red}{\enclose{circle}{3}}
    & \leftarrow\color{blue}{\enclose{circle}{4}}
    & \leftarrow\color{blue}{\enclose{circle}{5}}
    & \color{red}{6\phantom{,}}\\
\end{array}
\end{aligned}
$$

**Wrong Rule 2: A slice works like `range()`.** There are many similarities
between the behaviors of slices and the behavior of `range()`. However, they
do not behave the same. A slice
`start:end:step` only acts like `range(start, end, step)` if `start` and `end`
are **nonnegative**. If either of them are negative, the slice wraps around
and slices from the end of the array (see [below](negative-indices)).
`range()` on the other hand treats negative numbers as the actual start of end
values for the range. For example:

```py
>>> list(range(3, 5))
[3, 4]
>>> a[3:5] # a is range(7), and these are the same
[3, 4]
>>> list(range(3, -2)) # Empty, because -2 is less than 3
[]
>>> a[3:-2] # Indexes from 3 to the second to last (5)
[3, 4]
```

**Wrong Rule 3: Slices count the spaces between the elements of the array.**
This is a very common rule that is taught for both slices and integer
indexing. The reasoning goes as follows: 0-based indexing is confusing, where
the first element of an array is indexed by 0, the second by 1, and so on.
Rather than thinking about that, consider the spaces between the elements:

$$
\require{enclose}
\begin{aligned}
\begin{array}{c}
\begin{array}{r r r r r r r r r r r r r r r r r r}
a = & [&\phantom{|}&0, &\phantom{|} &1, & \phantom{|}& 2, &\phantom{|} & 3, &\phantom{|} &
4, &\phantom{|}& 5, &\phantom{|} & 6 &\phantom{|} &] &\\
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}\\
\color{red}{\text{index}}
    &
    & \color{red}{0}
    &
    & \color{red}{1}
    &
    & \color{red}{2}
    &
    & \color{red}{3}
    &
    & \color{red}{4}
    &
    & \color{red}{5}
    &
    & \color{red}{6}
    &
    & \color{red}{7}\\
\end{array}\\
\small{\text{(not a great way of thinking about indexes)}}
\end{array}
\end{aligned}
$$

Using this way of thinking, the first element of the array is to the left of
the "1-divider". An integer index `i` produces the element to the right of the
"`i`-divider", and a slice `i:j` picks the elements between the `i` and `j`
dividers.

At first glance, this seems like a rather clever way to think about the
half-open rule. For instance, between the `3` and `5` dividers is the subarray
`[3, 4]`, which is indeed what we get for `a[3:5]`. However, there are several
reasons why this way of thinking creates more confusion than it removes.

- As with wrong rule 1, it works well enough if the step is positive, but
  falls apart when it is negative.

  Consider again the slice `a[5:3:-1]`. Looking at the above figure, we might
  imagine it to give the same incorrect sub-array that we imagined before.


  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{c}
  \begin{array}{r r r r r r r r r r r r r r r r r r}
  a = & [&\phantom{|}&0, &\phantom{|} &1, & \phantom{|}& 2, &\phantom{|} & 3, &\phantom{|} &
  4, &\phantom{|}& 5, &\phantom{|} & 6 &\phantom{|} &] &\\
      &
      & \color{red}{|}
      &
      & \color{red}{|}
      &
      & \color{red}{|}
      &
      & \color{blue}{|}
      &
      & \color{blue}{|}
      &
      & \color{blue}{|}
      &
      & \color{red}{|}
      &
      & \color{red}{|}\\
  \color{red}{\text{index}}
      &
      & \color{red}{0}
      &
      & \color{red}{1}
      &
      & \color{red}{2}
      &
      & \color{blue}{3}
      &
      & \color{blue}{4}
      &
      & \color{blue}{5}
      &
      & \color{red}{6}
      &
      & \color{red}{7}\\
  \end{array}\\
  \small{\color{red}{\text{THIS IS WRONG!}}}
  \end{array}
  \end{aligned}
  $$

  As before, we might assume we would get

  ```py
  >> a[5:3:-1]
  [4, 3] # WRONG
  ```

  but this is incorrect! What we really get is

  ```py
  >>> a[5:3:-1]
  [5, 4]
  ```

- The rule does work for negative start and step, but only if you think about
  it correctly. The correct way to think about it is to reverse the indices

$$
\require{enclose}
\begin{aligned}
\begin{array}{c}
\begin{array}{r r r r r r r r r r r r r r r r r r}
a = & [&\phantom{|}&0, &\phantom{|} &1, & \phantom{|}& 2, &\phantom{|} & 3, &\phantom{|} &
4, &\phantom{|}& 5, &\phantom{|} & 6 &\phantom{|} &] &\\
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}\\
\color{red}{\text{index}}
    &
    & \color{red}{-7}
    &
    & \color{red}{-6}
    &
    & \color{red}{-5}
    &
    & \color{red}{-4}
    &
    & \color{red}{-3}
    &
    & \color{red}{-2}
    &
    & \color{red}{-1}
    &
    & \color{red}{0}\\
\end{array}\\
\small{\text{(not a great way of thinking about negative indexes)}}
\end{array}
\end{aligned}
$$

For example, `a[-4:-2]` will give `[3, 4]`

```py
>>> a[-4:-2]
[3, 4]
```

However, it would be quite easy to get confused here, as the "other" way of
thinking about negative indices (the way I am recommending) is that the end
starts at -1. So you might mistakenly imagine


$$
\require{enclose}
\begin{aligned}
\begin{array}{c}
\begin{array}{r r r r r r r r r r r r r r r r r r}
a = & [&\phantom{|}&0, &\phantom{|} &1, & \phantom{|}& 2, &\phantom{|} & 3, &\phantom{|} &
4, &\phantom{|}& 5, &\phantom{|} & 6 &\phantom{|} &] &\\
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}
    &
    & \color{red}{|}\\
\color{red}{\text{index}}
    &
    & \color{red}{-8}
    &
    & \color{red}{-7}
    &
    & \color{red}{-6}
    &
    & \color{red}{-5}
    &
    & \color{red}{-4}
    &
    & \color{red}{-3}
    &
    & \color{red}{-2}
    &
    & \color{red}{-1}\\
\end{array}\\
\small{\color{red}{\text{THIS IS WRONG!}}}
\end{array}
\end{aligned}
$$

- The rule "works" for slices, but is harder to imagine for integer indices.
  The integer index corresponding to the dividers corresponds to the entry to
  the *right* of the divider. Rules that involve remembering left or right
  aren't great for the memory.

- This rule leads to off-by-one errors due to "fencepost" errors. The
  fencepost problem is this: say you want to build a fence that is 100 feet
  long with posts spaced every 10 feet. How many fenceposts do you need? The
  naive answer is 10, but the correct answer is 11, because the fenceposts go
  in between the 10 feet divisions, including at the ends.

  <!-- TODO: Find an image to include here -->

  Fencepost problems are a leading cause of off-by-one errors. Thinking about
  slices in this way is to think about arrays as separated by fenceposts, and
  is only begging for problems. This will especially be the case if you still
  find yourself otherwise thinking about the indices of array elements
  themselves, rather than the divisions between them. And given the behavior
  of negative slices and integer indices under this model, one can hardly
  blame you for doing so.

Rather than trying to think about dividers between elements, it's much simpler
to just think about the elements themselves, but being counted with 0-based
indexing. 0-based indexing itself leads to off-by-one errors, since it is not
the usually way humans are taught to count things, but these will be far
fewer, especially as you gain practice in counting that way. As long as you
apply the rule "the end is not included", you will get the correct results.

**Wrong Rule 4: The `end` of a slice `a[start:end]` is 1-based.**

You might get clever and say `a[3:5]` indexes from the 3-rd element with
0-based indexing to the 5-th element with 1-based indexing. Don't do this. It
is confusing. Not only that, but the rule must necessarily be reversed for
negative indices. `a[-5:-3]` indexes from the -5-th element with -1-based
indexing to the -3-rd element with 0-based indexing (and of course, negative
and nonnegative starts and ends can be mixed, like `a[-5:5]`). Don't get cute
here. It isn't worth it.

(negative-indices)=
### Negative Indexes

(clipping)=
### Clipping

### Steps

### Negative Steps

(omitted)=
### Omitted Entries (`None`)
