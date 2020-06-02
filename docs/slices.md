Slices
======

Python's slice syntax is one of the more confusing parts of the language, even
to experienced developers. In this page, I carefully break down the rules for
slicing, and examine just what it is that makes it so confusing.

There are two primary aspects of slices that make them confusing:  confusing
conventions, and branching definitions. By confusing conventions, I mean that
slice semantics have definitions that are often difficult to reason about
mathematically. These conventions were chosen for syntactic convenience, and
one can easily see for most of them how they lead to concise notation for very
common operations, but it remains nonetheless true that they can make figuring
out the *right* slice to use in the first place complicated. By branching
definitions, I mean that the definition of a slice takes on fundamentally
different meanings if the start, end, or step are negative, nonnegative, or
omitted. This again is done for syntactic convenience, but it means that as a
user, you must switch your mode of thinking about slices depending on value of
the arguments. There is no uniform formula that applies to all slices.

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

What is a slice?
----------------

In Python, a slice is a special syntax that is allowed only in an index, that
is, inside of square brackets proceeding an expression. A slice consists of
one or two colons, with either an expression or nothing on either side of each
colon. For example, the following are all valid slices on the object `a`:

    a[x:y]
    a[x:y:z]
    a[:]
    a[x::]
    a[x::z]

Furthermore, for a slice `x:y:z` on Python or NumPy objects, there is an
additional semantic restriction, which is that the expressions `x`, `y`, and
`z` must be integers.

It is worth mentioning that the `x:y:z` syntax is not valid outside of square
brackets, but slice objects can be created manually using the `slice` builtin.
You can also use the `ndindex.Slice` object if you want to perform more
advanced operations. The discussions below will just use `x:y:z` without the
square brackets for simplicity.
<!-- TODO: Remove this? -->

(integer-indices)=
Integer indices
---------------

To understand slices, it is good to first review how integer indices work.
Throughout this guide, I will use as an example this prototype list:

<!-- TODO: Use a different list where the entries don't match the indices? -->

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

For **negative** integers, the indices index from the end of the array. These
indices are necessarily 1-based (or rather, -1-based), since 0 already refers
to the first element of the array. `-1` chooses the last element, `-2` the
second-to-last, and so on. For example, `a[-3]` picks the **third-to-last**
element of `a`, in this case, `4`:

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

The full definition of a slice could be written down in a couple of sentences,
although the branching definitions would necessitate several "if" conditions.
The [NumPy docs](https://numpy.org/doc/stable/reference/arrays.indexing.html)
on slices say

(numpy-definition)=

> The basic slice syntax is `i:j:k` where *i* is the starting index, *j* is
> the stopping index, and *k* is the step ( $k\neq 0$ ). This selects the `m`
> elements (in the corresponding dimension) with index values *i, i + k, ...,
> i + (m - 1) k* where $m = q + (r\neq0)$ and *q* and *r* are the quotient and
> remainder obtained by dividing *j - i* by *k*: *j - i = q k + r*, so that
> *i + (m - 1) k \< j*.

While notes like this may give a technically accurate description of slices,
they aren't especially helpful to someone who is trying to construct a slice
from a higher level of abstraction such as "I want to select this particular
subset of my array".

Instead, we shall examine slices by carefully going over all the various
aspects of the syntax and semantics that can lead to confusion, and attempting
to demystify them through simple rules.

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

For the slice `start:end`, with `start` and `end` nonnegative integers, the
indexes `start` and `end` are 0-based, just as with [integer
indexing](integer-indices) (although one should be careful that even though
`end` is 0-based, it is not included in the slice. See [below](half-open)).

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
omitted, which always slices to the beginning or end of the array, see
[below](omitted)).

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
advantages:

(sanity-check)=
- The maximum length of a slice `start:end`, when `start` and `end` are
  nonnegative, is always `end - start` (the caveat "maximum" is here because
  if `end` extends beyond the end of the array, then `start:end` will only
  slice up to `len(a) - start`, see [below](clipping)). For example, `a[i:i+n]`
  will slice `n` elements from the array `a`. Also be careful that this is
  only true when `start` and `end` are nonnegative (see
  [below](nonnegative-indices)). However, given those caveats, this is often a
  very useful sanity check that a slice is correct. If you expect a slice to
  have length `n` but `end - start` is clearly different from `n`, then the
  slice is likely wrong. Length calculations are more complicated when `step
  != 1`; in those cases, `len(ndindex.Slice(...))` can be useful.

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

**The proper rule to remember for half-open semantics is "the end is not
included".**

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
practice. You might as well think about slices using the [definition from the
NumPy docs](numpy-definition).

Rather, it is best to remember the simplest rule possible that is *always*
correct. That rule is, "the end is not included". That is always right,
regardless of what the values of `start`, `end`, or `step` are. The only
exception is if `end` is omitted. In this case, the rule obviously
doesn't apply as-is, and so you can fallback to the next rule about omitted
start/end (see [below](omitted)).

(wrong-rule-1)=
**Wrong Rule 1: "a slice `a[start:end]` slices the half-open interval
$[\text{start}, \text{end})$ (equivalently, a slice `a[start:end]` picks the
elements `i` such that `start <= i < end`)."** This is *only* the case if the
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
    &
    & )
    & \\
\end{array}\\
\small{\text{(reversed)}}\phantom{5,\quad 6]}
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

**Wrong Rule 2: "A slice works like `range()`."** There are many similarities
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

This rule is tempting because `range()` makes some computations easy. For
example, you can index or take the `len()` of a range. If you want to perform
computations on slices, I recommend using ndindex. This is what it was
designed for.

**Wrong Rule 3: "Slices count the spaces between the elements of the array."**
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

**Wrong Rule 4: "The `end` of a slice `a[start:end]` is 1-based."**

You might get clever and say `a[3:5]` indexes from the 3-rd element with
0-based indexing to the 5-th element with 1-based indexing. Don't do this. It
is confusing. Not only that, but the rule must necessarily be reversed for
negative indices. `a[-5:-3]` indexes from the -5-th element with -1-based
indexing to the -3-rd element with 0-based indexing (and of course, negative
and nonnegative starts and ends can be mixed, like `a[-5:5]`). Don't get cute
here. It isn't worth it.

(negative-indices)=
### Negative Indexes

Negative indices in slices work the same way they do with [integer
indices](integer-indices). For `start:end:step`, `start` and `end` cause the
indexing to go from the end of the array. However, they do not change the
direction of the slicing---only the `step` does that. The other rules of
slicing do not change when the `start` or `end` is negative. [The `end` is
still not included](half-open), values less than `-len(a)` still
[clip](clipping), and so on.

Note that positive and negative indices can be mixed. The following slices of
`a` all produce `[3, 4]`:

```py
>>> a[3:5]
[3, 4]
>>> a[-4:-2]
[3, 4]
>>> a[3:-2]
[3, 4]
>>> a[-4:5]
[3, 4]
```

$$
\begin{aligned}
\begin{array}{r r r r r r r r}
a = & [0, & 1, & 2, & 3, & 4, & 5, & 6]\\
\color{red}{\text{nonnegative index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{3\phantom{,}}
    & \color{blue}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{red}{6\phantom{,}}\\
\color{red}{\text{negative index}}
    & \color{red}{-7\phantom{,}}
    & \color{red}{-6\phantom{,}}
    & \color{red}{-5\phantom{,}}
    & \color{blue}{-4\phantom{,}}
    & \color{blue}{-3\phantom{,}}
    & \color{red}{-2\phantom{,}}
    & \color{red}{-1\phantom{,}}\\
\end{array}
\end{aligned}
$$

If a negative `end` indexes an element on or before a nonnegative `start`, the
slice is empty, the same as if `end <= start` when both are nonnegative.

```py
>>> a[3:-5]
[]
>>> a[3:2]
[]
```

Similar to integer indexes, negative indices `-i` in slices can always be
replaced by adding `len(a)` to `-i` until it is in the range $[0,
\operatorname{len}(a))$ (replacing `len(a)` with the size of the given axis
for NumPy arrays), so they are primarily a syntactic convenience.

The negative indexing behavior is convenient, but it can also lead to subtle
bugs, due to the fundamental discontinuity it produces. This is especially
likely to happen if the slice entries are arithmetical expressions. **One
should always double check if the `start` or `end` values of a slice can be
negative, and if they can, if those values produce the correct results.**

(negative-indices-example)=
For example, say you wanted to slice `n` values from the middle of `a`.
Something like the following would work

```py
>>> midway = len(a)//2
>>> n = 4
>>> a[midway - n//2: midway + n//2]
[1, 2, 3, 4]
```

From our [sanity check](sanity-check), `midway + n//2 - (midway - n//2)` does
equal `n` if `n` is even (we could find a similar expression for `n` odd, but
for now let us assume `n` is even).

However, let's look at what happens when `n` is larger than the size of `a`:

```py
>>> n = 8
>>> a[midway - n//2: midway + n//2]
[6]
```

This is mostly likely not what we would want. Depending on our use-case, we
would most likely want either an error or the full list `[0, 1, 2, 3, 4, 5,
6]`.

What happened here? Let's look at the slice values:

```py
>>> midway - n//2
-1
>>> midway + n//2
7
```

The `end` slice value is out of bounds for the array, but this just causes it
to [clip](clipping) to the end.

But `start` contains a subtraction, which causes it to become negative. Rather
than clipping to the start, it indexes from the end of the array, producing
the slice `a[-1:7]`. This picks the elements from the last element (`6`) up to
but not including the 7th element (0-based). Index 7 is out of bounds for the
array, so this picks all elements after `6`, which in this case is just `[6]`.

Unfortunately, the "correct" fix here depends on the desired behavior of each
individual slice. In some cases, the "slice from the end" behavior of negative
values is in fact what is desired. In others, you might prefer an error, so
should add a value check or assertion. In others, you might want clipping, in
which case you could modify the expression to always be nonnegative. For
example, instead of using `midway - n//2`, we could use `max(midway - n//2,
0)`.

```py
>>> n = 8
>>> a[max(midway - n//2, 0): midway + n//2]
[0, 1, 2, 3, 4, 5, 6]
```

(clipping)=
### Clipping

Slices can never give an out-of-bounds `IndexError`. This is different from
[integer indices](integer-indices) which require the index to be in bounds. If
`start` indexes before the beginning of the array (with a negative index), or
`end` indexes past the end of the array, the slice will clip to the bounds of
the array:

```py
>>> a[-100:100]
[0, 1, 2, 3, 4, 5, 6]
```

Furthermore, if the `start` is on or after the `end`, the slice will slice be
empty.

```py
>>> a[3:3]
[]
>>> a[5:2]
[]
```

For NumPy arrays, a consequence of this is that a slice will always keep the
axis, even if the size of the resulting axis is 0 or 1.

```py
>>> import numpy as np
>>> arr = np.array([[1, 2], [3, 4]])
>>> arr[0].shape # Removes the first dimension
(2,)
>>> arr[0:1].shape # Preserves the first dimension
(1, 2)
>>> arr[0:0].shape # Preserves the first dimension as an empty dimension
(0, 2)
```

An important consequence of the clipping behavior of slices is that you cannot
rely on runtime checks for out-of-bounds slices. See the [example
above](negative-indices-example). Another consequence is that you can never
rely on the length of a slice being `end - start` (for `step = 1` and `start`,
`end` nonnegative). This is rather the *maximum* length of the slice. It could
end up slicing something smaller. For example, an empty list will always slice
to an empty list. ndindex can help in calculations here:
`len(ndindex.Slice(...))` can be used to compute the *maximum* length of a
slice. If the shape of the input is known,
`len(ndindex.Slice(...).reduce(shape))` will compute the true length of the
slice.

### Steps

Thus far, we have only considered slices with the default step size of 1. When
the step is greater than 1, the slice picks every `step` element contained in
the bounds of `start` and `end`.

**The proper way to think about `step` is that the slice starts at `start` and
successively adds `step` until it reaches an index that is at or past the
`end`, and then stops without including that index.**

The important thing to remember about the `step` is that it being non-1 does
not change the fundamental rules of slices that we have learned so far.
`start` and `end` still use 0-based indexing. The `start` is always included
in the slice and the `end` is never included. Negative `start` and `end` index
from the end of the array. Out-of-bounds `start` and `end` still clip to the
beginning or end of the array.

Let us consider an example where the step size is `3`.

```py
>>> a[0:6:3]
[0, 3]
```

$$
\require{enclose}
\begin{aligned}
\begin{array}{r r r r r r r l}
a = & [0, & 1, & 2, & 3, & 4, & 5, &\ 6]\\
\color{red}{\text{index}}
    & \color{blue}{\enclose{circle}{0}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{\enclose{circle}{3}}
    & \color{red}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{red}{\enclose{circle}{6}}\\
    & \color{blue}{\text{start}}
    &
    & \rightarrow
    & \color{blue}{+3}
    &
    & \rightarrow
    & \color{red}{+3\ (\geq \text{end})}
\end{array}
\end{aligned}
$$

Note that the `start`, `0`, is included, but the `end`, `6`, is *not*
included, even though it is a multiple of `3` away from the start. This is
because the `end` is never included.

It can be tempting to think about the `step` in terms of modular arithmetic.
In fact, it is often the case in practice that you require a `step` greater
than 1 because you are dealing with modular arithmetic in some way. However,
this requires care.

Indeed, we can note that resulting indices `0`, `3` of the above slice
`a[0:6:3]` are all multiples of 3. This is because the `start` index, `0`, is
a multiple of 3. If we instead choose a start index that is $1 \pmod{3}$ then
all the indices would also be $1 \pmod{3}$.

```py
>>> a[1:6:3]
[1, 4]
```

However, be careful as this rule is *only* true for nonnegative `start`. If
`start` is negative, the value of $\text{start} \pmod{\text{step}}$ has no
bearing on the indices chosen for the slice:

```py
>>> list(range(21))[-15::3]
[6, 9, 12, 15, 18]
>>> list(range(22))[-15::3]
[7, 10, 13, 16, 19]
```

In the first case, `-15` is divisible by 3 and all the indices chosen by the
slice `-15::3` were also divisible by 3 (remember that the index and the value
are the same for simple ranges). But this is only because the length of the
list, `21`, also happened to be a multiple of 3. In the second example it is
`22` and the resulting indices are not multiples of `3`.

However, be aware that if the start is [clipped](clipping), the clipping
occurs *before* the step. That is, if the `start` is less than `len(a)`, it is
the same as `start = 0` regardless of the `step`.

```py
>>> a[-100::2]
[0, 2, 4, 6]
>>> a[-101::2]
[0, 2, 4, 6]
```

If you need to think about steps in terms of modular arithmetic,
`ndindex.Slice` can be used to perform various slice calculations so that you
don't have to come up with modulo formulas yourself. If you try to write such
formulas yourself, chances are you will get them wrong, as it is easy to fail
to properly account for negative vs. nonnegative indices, clipping, and
[negative steps](negative-steps). As was noted before, any correct "formula"
regarding slices will necessarily have many piecewise conditions.

(negative-steps)=
### Negative Steps

Recall what I said above:

**The proper way to think about `step` is that the slice starts at `start` and
successively adds `step` until it reaches an index that is at or past the
`end`, and then stops without including that index.**

The key thing to remember with negative `step` is that this rule still
applies. That is, the index starts at `start` then adds the `step` (which
makes the index smaller), and stops when it is at or past the `end`. Note the
phrase "at or past". If the `step` is positive this means "greater than or
equal to", but if the step is negative this means "less than or equal to".

Think of the step as starting at the `start` and sliding along the array,
jumping along by `step` spitting out elements. Once you see that you are at or
have gone past the `end` in the direction you are going (left for negative
`step` and right for positive `step`), you stop.

It's worth pointing out that unlike all other slices we have seen so far, a
negative `step` reverses the order that the elements are returned relative to
the original list. In fact, one of the most common uses of a negative step is
`a[::-1]`, which reverses the list:

```py
>>> a[::-1]
[6, 5, 4, 3, 2, 1, 0]
```

It is tempting therefore to think of a negative `step` as a "reversing"
operation. However, this is a bad way of thinking about negative steps. The
reason is that `a[i:j:-1]` is *not* equivalent to `reversed(a[j:i:1])`. The
reason is basically the same as was described in [wrong rule 1](wrong-rule-1)
above. The issue is that for `start:end:step`, `end` is *always* the what is
not included (see the [half-open](half-open) section above). Which means if we
swap `i` and `j`, we go from "`j` is not included" to "`i` is not included",
producing a wrong result. For example, as before:

```py
>>> a[5:3:-1]
[5, 4]
>>> list(reversed(a[3:5:1])) # This is not the same thing
[4, 3]
```

In the first case, index `3` is not included. In the second case, index `5` is
not included.

Worse, this way of thinking may even lead one to imagine the completely wrong
idea that `a[i:j:-1]` is the same as `reversed(a)[j:i]`:

```py
>>> list(reversed(a))[3:5]
[3, 2]
```

Once `a` is reversed, the indices `3` and `5` have nothing to do with the
original indices `3` and `5`. To see why, consider a much larger list:

```py
>>> list(range(100))[5:3:-1]
[5, 4]
>>> list(reversed(range(100)))[3:5]
[96, 95]
```

It is much more robust to think about the slice as starting at `start`, then
moving across the list by `step` until reaching `end`, which is not included.

Negative steps can of course be less than -1 as well, with similar behavior to
steps greater than 1, again, keeping in mind that the end is not included.

```py
>>> a[6:0:-3]
[6, 3]
```

$$
\require{enclose}
\begin{aligned}
\begin{array}{r r r r r r r r l}
a = & [0, & 1, & 2, & 3, & 4, & 5, &\ 6]\\
\color{red}{\text{index}}
    & \color{red}{\enclose{circle}{0}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{\enclose{circle}{3}}
    & \color{red}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{blue}{\enclose{circle}{6}}\\
    & \color{red}{-3}
    &
    & \leftarrow
    & \color{blue}{-3}
    &
    & \leftarrow
    & \color{blue}{\text{start}}\\
    &  (\leq \text{end})
\end{array}
\end{aligned}
$$

The `step` can never be equal to 0. This unconditionally leads to an error:

```py
>>> a[::0]
Traceback (most recent call last):
...
ValueError: slice step cannot be zero
```

(omitted)=
### Omitted Entries

The final point of confusion is omitted entries.[^ommited-none]

[^ommited-none]: `start`, `end`, or `step` may also be `None`, which is
syntactically equivalent to them being omitted. That is to say, `a[::]` is a
syntax shorthand for `a[None:None:None]`. It is rare to see `None` in a slice;
this is only relevant for code that consumes slices, such as a `__getitem__`
method on an object. The `slice` object corresponding to a slice `a[::]` is
`slice(None, None, None)`. `ndindex.Slice()` also uses `None` to indicate
omitted entries in the same way.

**The best way to think about omitted entries is just like that, as omitted
entries.** That is, for a slice like `a[:i]` think of it as the `start` being
omitted, and `end` equal to `i`. Conversely, `a[i:]` has the `start` as `i`
and the `end` omitted. The wrong way to think about these is as a colon being
before or after the index `i`. Thinking about it this way will only lead to
confusion, because you won't be thinking about `start` and `end`, but rather
trying to remember some rule based on where a colon is. But the colons in a
slice are not indicators, they are separators.

As to the semantic meaning of omitted entries, the easiest one is the `step`.
If the `step` is omitted, it always defaults to `1`. If the step is omitted
the second colon before the step can also be omitted. That is to say, the
following are completely equivalent:

```py
a[i:j:1]
a[i:j:]
a[i:j]
```

<!-- TODO: Better wording for this rule? -->
For the `start` and `end`, the rule is that being omitted extends the slice
all the way to the edge of the list in the direction being sliced. If the
`step` is positive, this means `start` extends to the beginning of the list
and `end` extends to the end. If `step` is negative, it is reversed: `start`
extends to the end of the array and `end` extends to the beginning.

## Rules

These rules are the ones to keep in mind to understand how slices work. For a
slice `a[start:end:step]`:

1. `start` and `step` use 0-based indexing from the start of the array when
   they are nonnegative, and -1-based indexing from end of the array when they
   are negative.
2. `end` is never included in the slice.
3. `start` and `end` are clipped to the bounds of the array.
4. The slice starts at `start` and successively adds `step` until it reaches
   an index that is at or past `end`, and then stops without including that
   `end` index.
5. If `step` is omitted it defaults to 1.
6. If `start` or `end` are omitted they extend to the start or end of the
   array in the direction being sliced. Slices like `a[:i]` or `a[i:]` should
   be though of as the `start` or `end` being omitted, not as a colon to the
   left or right of an index.
7. Slicing something never produces an `IndexError`, even if the slice is
   empty. For a NumPy array, a slice always keeps the axis being sliced, even
   if the final dimension is 0 or 1.
8. These rules make it syntactically convenient to slice subarrays in useful
   ways, but make it extremely challenging to write down formulas for things
   corresponding to slices that are correct in all cases. Instead of trying to
   do this yourself, use ndindex.

# Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
