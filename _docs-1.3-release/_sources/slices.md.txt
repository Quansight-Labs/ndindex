(slices-docs)=
Slices
======

Python's slice syntax is one of the more confusing parts of the language, even
to experienced developers. In this page, I carefully break down the rules for
slicing, and examine just what it is that makes it so confusing.

There are two primary aspects of slices that make them difficult to
understand: confusing conventions, and discontinuous definitions. By confusing
conventions, we mean that slice semantics have definitions that are often
difficult to reason about mathematically. These conventions were chosen for
syntactic convenience, and one can easily see for most of them how they lead
to concise notation for very common operations, but it remains nonetheless
true that they can make figuring out the *right* slice to use in the first
place complicated. By discontinuous definitions, we mean that the definition
of a slice takes on fundamentally different meanings if the start, stop, or
step are negative, nonnegative, or omitted. This again is done for syntactic
convenience, but it means that as a user, you must switch your mode of
thinking about slices depending on value of the arguments. There are no
uniform formulas that apply to all slices.

The [ndindex](index) library can help with much of this, especially for people
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

```py
a[x:y]
a[x:y:z]
a[:]
a[x::]
a[x::z]
```

Furthermore, for a slice `a[x:y:z]`, if `a` is a Python built-in object or a
NumPy array, there is an additional semantic restriction, which is that the
expressions `x`, `y`, and `z` must be integers.

The three arguments to a slice are traditionally called `start`, `stop`, and
`step`:

```py
a[start:stop:step]
```

We will use these names throughout this guide.

It is worth noting that the `x:y:z` syntax is not valid outside of square
brackets. However, slice objects can be created manually using the `slice()`
builtin (`a[x:y:z]` is the same as `a[slice(x, y, z)]`). You can also use the
[`ndindex.Slice()`](slice-api) object if you want to perform more advanced
operations.

(rules)=
## Rules

These are the rules to keep in mind to understand how slices work. Each of
these is explained in detail below. Many of the detailed descriptions below
also outline several *wrong* rules, which are bad ways of thinking about
slices but which you may be tempted to think about as rules. The below 7 rules
are always correct.

In this document, "*nonnegative*" means $\geq 0$ and "*negative*" means $< 0$.

For a slice `a[start:stop:step]`:

1. `start` and `stop` use **0-based indexing** from the **beginning** of `a`
   when they are **nonnegative**, and **−1-based indexing** from **end** of
   `a` when they are **negative**. (See sections {ref}`0-based` and
   {ref}`negative-indices`)
2. `stop` is never included in the slice. (See section {ref}`half-open`)
3. `start` and `stop` are clipped to the bounds of `a`. (See section {ref}`clipping`)
4. The slice starts at `start` and successively adds `step` until it reaches
   an index that is at or past `stop`, and then stops without including that
   `stop` index. (See sections {ref}`steps` and {ref}`negative-steps`)
5. If `step` is omitted it defaults to `1`. (See section {ref}`omitted`)
6. If `start` or `stop` are omitted they extend to the beginning or end of `a`
   in the direction being sliced. Slices like `a[:i]` or `a[i:]` should be
   though of as the `start` or `stop` being omitted, not as a colon to the
   left or right of an index. (See section {ref}`omitted`)
7. Slicing something never raises an `IndexError`, even if the slice is empty.
   For a NumPy array, a slice always keeps the axis being sliced, even if that
   means the resulting dimension will be 0 or 1. (See section {ref}`subarray`)

(integer-indices)=
## Integer indices

To understand slices, it is good to first review how integer indices work.
Throughout this guide, we will use as an example this prototype list:

$$
a = [\mathtt{\textsf{'}a\textsf{'}},\ \mathtt{\textsf{'}b\textsf{'}},\ \mathtt{\textsf{'}c\textsf{'}},\
\mathtt{\textsf{'}d\textsf{'}},\ \mathtt{\textsf{'}e\textsf{'}},\ \mathtt{\textsf{'}f\textsf{'}},\ \mathtt{\textsf{'}g\textsf{'}}]
$$

The list `a` has 7 elements.

The elements of `a` are strings, but the indices and slices on the list `a`
will always use integers. An index or slice is never based on the value of the
elements, but rather the position of the elements in the list.[^dict-footnote]

[^dict-footnote]: If you are looking for something that allows non-integer
indices or that indexes by value, you may want a `dict`. Despite using similar
syntax, `dict`s do not allow slicing.

An integer index picks a single element from the list `a`. For NumPy arrays,
integer indices pick a subarray corresponding to a particular element from a
given axis (and as a result, an integer index always reduces the
dimensionality of an array by one).

(fourth-sentence)=
The key thing to remember about indexing in Python, both for integer and
slice indexing, is that it is 0-based. This means that the indices start
at 0. This is the case for all **nonnegative** indices.
For example, `a[3]` would pick the **fourth** element of `a`, in this case, `'d'`.

<div style="text-align:center">
<code style="font-size: 16pt;">a[3] == 'd'</code>
$$
\begin{aligned}
\begin{array}{r c c c c c c c}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
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
</div>

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[3]
'd'
```

0-based indexing is different from how people typically count things, which is
1-based (1, 2, 3, ...). Thinking in terms of 0-based indexing requires some
practice.

For **negative** integers, indices index from the end of the list. These
indices are necessarily 1-based (or rather, −1-based), since `0` already
refers to the first element of the list. `-1` chooses the last element, `-2`
the second-to-last, and so on. For example, `a[-3]` picks the
**third-to-last** element of `a`, in this case, `'e'`:

<div style="text-align:center">
<code style="font-size: 16pt;">a[-3] == 'e'</code>
$$
\begin{aligned}
\begin{array}{r c c c c c c c}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
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
</div>

```py
>>> a[-3]
'e'
```

An equivalent way to think about negative indices is that an index
`a[-i]` picks `a[len(a) - i]`, that is, you can subtract the negative
index off of the size of `a` (for a NumPy array, replace `len(a)`
with the size of the axis being sliced). For example, `len(a)` is `7`, so
`a[-3]` is the same as `a[7 - 3]`:

```py
>>> len(a)
7
>>> a[7 - 3]
'e'
```

Therefore, negative indices are primarily a syntactic convenience that
allows one to specify parts of a list that would otherwise need to be
specified in terms of the size of the list.

If an integer index is greater than or equal to the size of the list, or less
than negative the size of the list (`i >= len(a)` or `i < len(a)`), then it
is out of bounds and will raise an `IndexError`.

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

## Points of Confusion

Now let us come back to slices. The full definition of a slice could be
written down in a couple of sentences, although the discontinuous definitions
would necessitate several "if" conditions. The [NumPy
docs](https://numpy.org/doc/stable/reference/arrays.indexing.html) on slices
say

(numpy-definition)=

> The basic slice syntax is `i:j:k` where *i* is the starting index, *j* is
> the stopping index, and *k* is the step ( $k\neq 0$ ). This selects the `m`
> elements (in the corresponding dimension) with index values *i, i + k, ...,
> i + (m - 1) k* where $m = q + (r\neq 0)$ and *q* and *r* are the quotient and
> remainder obtained by dividing *j - i* by *k*: *j - i = q k + r*, so that
> *i + (m - 1) k \< j*.

While definitions like this may give a technically accurate description of
slices, they aren't especially helpful to someone who is trying to construct a
slice from a higher level of abstraction such as "I want to select this
particular subset of my array".[^numpy-definition-footnote]

[^numpy-definition-footnote]: This formulation actually isn't particularly
helpful for formulating higher level slice formulas such as the ones used by
ndindex either.

Instead, we shall examine slices by carefully going over all the various
aspects of the syntax and semantics that can lead to confusion, and attempting
to demystify them through simple [rules](rules).

(subarray)=
### Subarray

**A slice always produces a subarray (or sub-list, sub-tuple, sub-string,
etc.). For NumPy arrays, this means that a slice will always *preserve* the
dimension that is sliced.** This is true even if the slice chooses only a
single element, or even if it chooses no elements. This is also true for
lists, tuples, and strings, in the sense that a slice on a list, tuple, or
string will always produce a list, tuple, or string. This behavior is
different from integer indices, which always remove the dimension that they
index.

For example

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[3]
'd'
>>> a[3:4]
['d']
>>> a[5:2] # Empty slice
[]
>>> import numpy as np
>>> arr = np.array([[1, 2], [3, 4]])
>>> arr.shape
(2, 2)
>>> arr[0].shape # Integer index removes the first dimension
(2,)
>>> arr[0:1].shape # Slice preserves the first dimension
(1, 2)
```

One consequence of this is that, unlike integer indices, **slices will never
raise `IndexError`, even if the slice is empty**. Therefore you cannot rely on
runtime errors to alert you to coding mistakes relating to slice bounds that
are too large. A slice cannot be "out of bounds." See the section on
[clipping](#clipping) below.

(0-based)=
### 0-based

For the slice `a[start:stop]`, with `start` and `stop` nonnegative integers,
the indices `start` and `stop` are 0-based, just as with [integer
indexing](integer-indices) (although one should be careful that even though
`stop` is 0-based, it is not included in the slice, see [below](half-open)).

For example:

<div style="text-align:center">
<code style="font-size: 16pt;">a[3:5] == ['d', 'e']</code>
$$
\begin{aligned}
\begin{array}{r c c c c c c c}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
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
</div>

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[3:5]
['d', 'e']
```

Do not be worried if you find 0-based indexing hard to get used to, or if you
find yourself forgetting about it. Even experienced Python developers (this
author included) still find themselves writing `a[3]` instead of `a[2]` from
time to time. The best way to learn to use 0-based indexing is to practice
using it enough that you use it automatically without thinking about it.

(half-open)=
### Half-open

Slices behave like half-open intervals. What this means is that the `stop` in
`a[start:stop]` is *never* included in the slice (the exception is if the
`stop` is omitted, see [below](omitted)).

For example, `a[3:5]` slices the indices `3` and `4`, but not `5`
([0-based](0-based)).

<div style="text-align:center">
<code style="font-size: 16pt;">a[3:5] == ['d', 'e']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r c c c c c c c}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
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
</div>

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[3:5]
['d', 'e']
```

The half-open nature of slices means that you must always remember that the
`stop` slice element is not included in the slice. However, it has a few
advantages:

(sanity-check)=
- The maximum length of a slice `a[start:stop]`, when `start` and `stop` are
  nonnegative, is always `stop - start`. For example, `a[i:i+n]` will slice
  `n` elements from `a`. The caveat "maximum" is here because if `stop`
  extends beyond the end of `a`, then `a[start:stop]` will only slice up
  to `len(a) - start` (see {ref}`clipping` below). Also be careful that this
  is only true when `start` and `stop` are nonnegative (see
  {ref}`negative-indices` below). However, given those caveats, this is often
  a very useful sanity check that a slice is correct. If you expect a slice to
  have length `n` but `stop - start` is clearly different from `n`, then the
  slice is likely wrong. Length calculations are more complicated when `step
  != 1`; in those cases, {meth}`len(ndindex.Slice(...)) <ndindex.Slice.__len__>` can be useful.

- `len(a)` can be used as a `stop` value to slice to the end of `a`. For
  example, `a[1:len(a)]` slices from the second element to the end of `a`
  (this is equivalent to `a[1:]`, see {ref}`omitted`)

  ```py
  >>> a[1:len(a)]
  ['b', 'c', 'd', 'e', 'f', 'g']
  >>> a[1:]
  ['b', 'c', 'd', 'e', 'f', 'g']
  ```

- Consecutive slices can be appended to one another by making each successive
  slice's `start` the same as the previous slice's `stop`. For example, for our
  list `a`, `a[2:3] + a[3:5]` is the same as `a[2:5]`.

  ```py
  >>> a[2:3] + a[3:5]
  ['c', 'd', 'e']
  >>> a[2:5]
  ['c', 'd', 'e']
  ```

  A common usage of this is to split a slice into two slices. For example, the
  slice `a[i:j]` can be split as `a[i:k]` and `a[k:j]`.

#### Wrong Ways of Thinking about Half-open Semantics

**The proper rule to remember for half-open semantics is "the `stop` is not
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
correct. That rule is, "the `stop` is not included". This rule is extremely
simple, and is always right, regardless of what the values of `start`, `stop`,
or `step` are. The only exception is if `stop` is omitted. In this case, the
rule obviously doesn't apply as-is, and so you can fallback to the rule about
omitted `start`/`stop` (see {ref}`omitted` below).

(wrong-rule-1)=
<strong style="font-size:120%;" style="font-size:120%;">Wrong Rule 1: "a slice `a[start:stop]` slices the half-open interval
$[\text{start}, \text{stop})$ (equivalently, a slice `a[start:stop]` picks the
elements `i` such that `start <= i < stop`)." <a class="headerlink"
href="#wrong-rule-1" title="Permalink to this headline">¶</a> </strong>

This is *only* the case if the `step` is positive. It also isn't directly true
for negative `start` or `stop`. For example, with a `step` of `-1`,
`a[start:stop:-1]` slices starting at `start` going in reverse order to
`stop`, but not including `stop`. Mathematically, this creates a half open
interval $(\text{stop}, \text{start}]$ (except reversed).

For example, say way believed that `a[5:3:-1]` sliced the half-open interval
$[3, 5)$ but in reverse order.

<!-- TODO: improve this -->
<div style="text-align:center;" >
<code style="font-size: 16pt;">a[5:3:-1] "==" ['e', 'd']</code>
<div style="font-size: 16pt;color:red;">(WRONG)</div>
$$
\begin{aligned}
\begin{array}{r c c c c c c c}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
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
\small{\text{(reversed)}}\hspace{3.5em}
\end{aligned}
$$
</div>

We might assume we would get

```py
>> a[5:3:-1]
['e', 'd'] # WRONG
```

Actually, what we really get is

```py
>>> a[5:3:-1]
['f', 'e']
```

This is because the slice `5:3:-1` starts at index `5` and steps backwards to
index `3`, but not including `3`.

<div style="text-align:center">
<code style="font-size: 16pt;">a[5:3:-1] == ['f', 'e']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r r r r r r r}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{\textsf{'},}}
    & \color{red}{1\phantom{\textsf{'},}}
    & \color{red}{2\phantom{\textsf{'},}}
    & \color{red}{\enclose{circle}{3}\phantom{,}}
    & \leftarrow\color{blue}{\enclose{circle}{4}\phantom{,}}
    & \leftarrow\color{blue}{\enclose{circle}{5}\phantom{,}}
    & \color{red}{6\phantom{\textsf{'},}}\\
\end{array}
\end{aligned}
$$
</div>

(wrong-rule-2)=
<strong style="font-size:120%;">Wrong Rule 2: "A slice works like `range()`." <a class="headerlink"
href="#wrong-rule-2" title="Permalink to this headline">¶</a> </strong>

There are many similarities between the behaviors of slices and the behavior
of `range()`. However, they do not behave the same. A slice
`a[start:stop:step]` only acts like `range(start, stop, step)` if `start` and
`stop` are **nonnegative**. If either of them are negative, the slice wraps
around and slices from the end of the list (see {ref}`negative-indices`
below). `range()` on the other hand treats negative numbers as the actual
start and stop values for the range. For example:

```py
>>> list(range(3, 5))
[3, 4]
>>> b = list(range(7))
>>> b[3:5] # b is range(7), and these are the same
[3, 4]
>>> list(range(3, -2)) # Empty, because -2 is less than 3
[]
>>> b[3:-2] # Indexes from 3 to the second to last (5)
[3, 4]
```

This rule is tempting because `range()` makes some computations easy. For
example, you can index or take the `len()` of a range. If you want to perform
computations on slices, we recommend using [ndindex](slice-api). This is what
it was designed for.

(wrong-rule-3)=
<strong style="font-size:120%;">Wrong Rule 3: "Slices index the spaces between the elements of the list."<a class="headerlink"
href="#wrong-rule-3" title="Permalink to this headline">¶</a> </strong>

This is a very common rule that is taught for both slices and integer
indexing. The reasoning goes as follows: 0-based indexing is confusing, where
the first element of a list is indexed by 0, the second by 1, and so on.
Rather than thinking about that, consider the spaces between the elements:

<div style="text-align:center">
<code style="font-size: 16pt;">a[3:5] == ['d', 'e']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r r r r r r r r r r r r r r r r r}
a = & [&\phantom{|}&\mathtt{\textsf{'}a\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}b\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}c\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}d\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}e\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}f\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}g\textsf{'}}&\phantom{|}&]&\\
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
\end{aligned}
$$
<i>(not a great way of thinking about indices)</i>
</div>

Using this way of thinking, the first element of `a` is to the left of
the "1-divider". An integer index `i` produces the element to the right of the
"`i`-divider", and a slice `a[i:j]` picks the elements between the `i` and `j`
dividers.

At first glance, this seems like a rather clever way to think about the
half-open rule. For instance, between the `3` and `5` dividers is the subarray
`['d', 'e']`, which is indeed what we get for `a[3:5]`. However, there are several
reasons why this way of thinking creates more confusion than it removes.

- As with [wrong rule 1](wrong-rule-1), it works well enough if the step is
  positive, but falls apart when it is negative.

  Consider again the slice `a[5:3:-1]`. Looking at the above figure, we might
  imagine it to give the same incorrect subarray that we imagined before.

  <div style="text-align:center">
  <code style="font-size: 16pt;">a[5:3:-1] "==" ['e', 'd']</code>
  <div style="font-size: 16pt;color:red;">(WRONG)</div>
  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{c}
  \begin{array}{r r r r r r r r r r r r r r r r r r}
    a = & [&\phantom{|}&\mathtt{\textsf{'}a\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}b\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}c\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}d\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}e\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}f\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}g\textsf{'}}&\phantom{|}&]&\\
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
  \small{\color{red}{\textbf{THIS IS WRONG!}}}
  \end{array}
  \end{aligned}
  $$
  </div>

  As before, we might assume we would get

  ```py
  >> a[5:3:-1]
  ['e', 'd'] # WRONG
  ```

  but this is incorrect! What we really get is

  ```py
  >>> a[5:3:-1]
  ['f', 'e']
  ```

  If you've ever espoused the "spaces between elements" way of thinking about
  indices, this should give you serious pause. One can see from the above
  diagram that there is clearly no way to salvage this rule to see `a[5:3:-1]`
  as giving `['f', 'e']`. Contrast the same slice when simply thinking about
  it as stepping backwards from index `5` to index `3`, but not including
  index `3`.

  <div style="text-align:center">
  <code style="font-size: 16pt;">a[5:3:-1] == ['f', 'e']</code>
  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{r r c c c c c c c c c c c c l}
  a = & [\mathtt{\textsf{'}a\textsf{'}}, && \mathtt{\textsf{'}b\textsf{'}}, && \mathtt{\textsf{'}c\textsf{'}}, && \mathtt{\textsf{'}d\textsf{'}}, && \mathtt{\textsf{'}e\textsf{'}}, && \mathtt{\textsf{'}f\textsf{'}}, && \mathtt{\textsf{'}g\textsf{'}}]\\
  \color{red}{\text{index}}
      & \color{red}{0\phantom{,}}
      &
      & \color{red}{1\phantom{,}}
      &
      & \color{red}{2\phantom{,}}
      &
      & \color{red}{\enclose{circle}{3}}
      &
      & \color{blue}{\enclose{circle}{4}}
      &
      & \color{blue}{\enclose{circle}{5}}
      &
      & \color{red}{6\phantom{,}}\\
      &
      & \phantom{\leftarrow}
      &
      & \phantom{\leftarrow}
      &
      & \phantom{\leftarrow}
      & \color{red}{-1}
      & \leftarrow
      & \color{blue}{-1}
      & \leftarrow
      & \color{blue}{\text{start}}
  \end{array}
  \end{aligned}
  $$
  </div>

- The rule does work for negative `start` and `stop`, but only if you think
  about it correctly. The correct way to think about it is to reverse the
  indices:

  <div style="text-align:center" >
  <code style="font-size: 16pt;">a[-4:-2] == ['d', 'e']</code>
  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{c}
  \begin{array}{r r r r r r r r r r r r r r r r r r}
    a = & [&\phantom{|}&\mathtt{\textsf{'}a\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}b\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}c\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}d\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}e\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}f\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}g\textsf{'}}&\phantom{|}&]&\\
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
      & \color{red}{-7}
      &
      & \color{red}{-6}
      &
      & \color{red}{-5}
      &
      & \color{blue}{-4}
      &
      & \color{blue}{-3}
      &
      & \color{blue}{-2}
      &
      & \color{red}{-1}
      &
      & \color{red}{0}\\
  \end{array}\\
  \small{\text{(not a great way of thinking about negative indices)}}
  \end{array}
  \end{aligned}
  $$
  </div>

  For example, `a[-4:-2]` will give `['d', 'e']`

  ```py
  >>> a[-4:-2]
  ['d', 'e']
  ```

  However, it would be quite easy to get confused here, as the "other" way of
  thinking about negative indices (the way we are recommending) is that the
  end starts at -1. So you might mistakenly imagine something like this:

  <div style="text-align:center" >
  <code style="font-size: 16pt;">a[-4:-2] "==" ['e', 'f']</code>
  <div style="font-size: 16pt;color:red;">(WRONG)</div>
  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{c}
  \begin{array}{r r r r r r r r r r r r r r r r r r}
    a = & [&\phantom{|}&\mathtt{\textsf{'}a\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}b\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}c\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}d\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}e\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}f\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}g\textsf{'}}&\phantom{|}&]&\\
      &
      & \color{red}{|}
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
      & \color{blue}{-4}
      &
      & \color{blue}{-3}
      &
      & \color{blue}{-2}
      &
      & \color{red}{-1}\\
  \end{array}\\
  \small{\color{red}{\textbf{THIS IS WRONG!}}}
  \end{array}
  \end{aligned}
  $$
  </div>


  But things are even worse than that. If we combine negative `start` and
  `stop` and negative `step`, things get even more confusing. Consider the
  slice `a[-2:-4:-1]`. This gives `['f', 'e']`:

  ```py
  >>> a[-2:-4:-1]
  ['f', 'e']
  ```

  To get this with the "spacers" idea, we have to use the above "wrong"
  diagram:

  <div style="text-align:center" >
  <code style="font-size: 16pt;">a[-2:-4:-1] == ['f', 'e']</code>
  <div style="font-size: 16pt;color:blue;">NOW RIGHT!</div>
  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{c}
  \begin{array}{r r r r r r r r r r r r r r r r r r}
    a = & [&\phantom{|}&\mathtt{\textsf{'}a\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}b\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}c\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}d\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}e\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}f\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}g\textsf{'}}&\phantom{|}&]&\\
      &
      & \color{red}{|}
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
      & \color{blue}{-4}
      &
      & \color{blue}{-3}
      &
      & \color{blue}{-2}
      &
      & \color{red}{-1}\\
  \end{array}\\
  \small{\text{(not a great way of thinking about negative indices)}}
  \end{array}
  \end{aligned}
  $$
  </div>

  <div style="text-align:center" >
  <code style="font-size: 16pt;">a[-2:-4:-1] "==" ['e', 'd']</code>
  <div style="font-size: 16pt;color:red;">(WRONG)</div>
  $$
  \require{enclose}
  \begin{aligned}
  \begin{array}{c}
  \begin{array}{r r r r r r r r r r r r r r r r r r}
    a = & [&\phantom{|}&\mathtt{\textsf{'}a\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}b\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}c\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}d\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}e\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}f\textsf{'}}, &\phantom{|}& \mathtt{\textsf{'}g\textsf{'}}&\phantom{|}&]&\\
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
      & \color{red}{-7}
      &
      & \color{red}{-6}
      &
      & \color{red}{-5}
      &
      & \color{blue}{-4}
      &
      & \color{blue}{-3}
      &
      & \color{blue}{-2}
      &
      & \color{red}{-1}
      &
      & \color{red}{0}\\
  \end{array}\\
  \small{\color{red}{\textbf{THIS IS WRONG!}}}
  \end{array}
  \end{aligned}
  $$
  </div>

  In other words, the "right" way to think of spacers with negative `start`
  and `stop` depends if the `step` is positive or negative.

  This is because the correct half-open rule is based on not including the
  `stop`. It *isn't* based on not including the larger end of the interval. If
  the `step` is positive, the `stop` will be larger, but if it is
  [negative](negative-steps), the `start` will be larger.

- The rule "works" for slices, but is harder to imagine for integer indices.
  In the divider way of thinking, an integer index `n` corresponds to the
  entry to the *right* of the `n` divider. Rules that involve remembering left
  or right aren't great when it comes to memorability.

(fencepost)=
- This rule leads to off-by-one errors due to "fencepost" errors. The
  fencepost problem is this: say you want to build a fence that is 100 feet
  long with posts spaced every 10 feet. How many fenceposts do you need?

  The naive answer is 10, but the correct answer is 11. The reason is the
  fenceposts go in between the 10 feet divisions, including at the end points.
  So there is an "extra" fencepost compared to the number of fence sections.


  ```{figure} imgs/jeff-burak-lPO0VzF_4s8-unsplash.jpg
  A section of a fence that has 6 segments and 7 fenceposts.[^fencepost-jeff-burbak-footnote]

  [^fencepost-jeff-burbak-footnote]: Image credit [Jeff Burak via
  Unsplash](https://unsplash.com/photos/lPO0VzF_4s8). The image is of
  Chautauqua Park in Boulder, Colorado.
  ```

  Fencepost problems are a leading cause of off-by-one errors. Thinking about
  slices in this way is to think about lists as separated by fenceposts, and
  is only begging for problems. This will especially be the case if you still
  find yourself otherwise thinking about indices as pointing to list elements
  themselves, rather than the divisions between them. And given the behavior
  of negative slices and integer indices under this model, one can hardly
  blame you for doing so (see the previous two bullet points).

Rather than trying to think about dividers between elements, it's much simpler
to just think about the elements themselves, but being counted starting at 0.
To be sure, 0-based indexing itself leads to off-by-one errors, since it is
not the usually way humans are taught to count things, but this is nonetheless
the best way to think about things, especially as you gain practice in
counting that way. As long as you apply the rule "the `stop` is not included,"
you will get the correct results.

(wrong-rule-4)=
<strong style="font-size:120%;">Wrong Rule 4: "The `stop` of a slice `a[start:stop]` is 1-based."<a class="headerlink"
href="#wrong-rule-4" title="Permalink to this headline">¶</a> </strong>

You might get clever and say `a[3:5]` indexes from the 3rd element with
0-based indexing to the 5th element with 1-based indexing. Don't do this. It
is confusing. Not only that, but the rule must necessarily be reversed for
negative indices. `a[-5:-3]` indexes from the (−5)th element with −1-based
indexing to the (−3)rd element with 0-based indexing (and of course, negative
and nonnegative starts and stops can be mixed, like `a[3:-3]`). Don't get cute
here. It isn't worth it.

(negative-indices)=
### Negative Indices

Negative indices in slices work the same way they do with [integer
indices](integer-indices). **For `a[start:stop:step]`, negative `start` or
`stop` use −1-based indexing from the end of `a`.** However, negative `start`
or `stop` does *not* change the order of the slicing---only the `step` does
that. The other [rules](rules) of slicing do not change when the `start` or
`stop` is negative. [The `stop` is still not included](half-open), values less
than `-len(a)` still [clip](clipping), and so on.

Note that positive and negative indices can be mixed. The following slices of
`a` all produce `['d', 'e']`:

<div style="text-align:center">
<div style="font-size: 16pt;"><code>a[3:5] == a[-4:-2] == a[3:-2] == a[-4:5]
== ['d', 'e']</code></div>
$$
\begin{aligned}
\begin{array}{r c c c c c c c}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{nonnegative index}}
    & \color{red}{0\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{\enclose{circle}{2}\phantom{,}}
    & \color{blue}{3\phantom{,}}
    & \color{blue}{4\phantom{,}}
    & \color{red}{\enclose{circle}{5}\phantom{,}}
    & \color{red}{6\phantom{,}}\\
\color{red}{\text{negative index}}
    & \color{red}{-7\phantom{,}}
    & \color{red}{-6\phantom{,}}
    & \color{red}{\enclose{circle}{-5}\phantom{,}}
    & \color{blue}{-4\phantom{,}}
    & \color{blue}{-3\phantom{,}}
    & \color{red}{\enclose{circle}{-2}\phantom{,}}
    & \color{red}{-1\phantom{,}}\\
\end{array}
\end{aligned}
$$
</div>

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[3:5]
['d', 'e']
>>> a[-4:-2]
['d', 'e']
>>> a[3:-2]
['d', 'e']
>>> a[-4:5]
['d', 'e']
```

If a negative `stop` indexes an element on or before a nonnegative `start`, the
slice is empty, the same as if `stop <= start` when both are nonnegative.

```py
>>> a[3:-5]
[]
>>> a[3:2]
[]
```

Similar to integer indices, negative indices `-i` in slices can always be
replaced by adding `len(a)` to `-i` until it is in the range $[0,
\operatorname{len}(a))$ (replacing `len(a)` with the size of the given axis
for NumPy arrays), so they are primarily a syntactic convenience.

The negative indexing behavior is convenient, but it can also lead to subtle
bugs, due to the fundamental discontinuity it produces. This is especially
likely to happen if the slice entries are arithmetical expressions. **One
should always double check if the `start` or `stop` values of a slice can be
negative, and if they can, if those values produce the correct results.**

(negative-indices-example)=
For example, say you wanted to slice `n` values from the middle of `a`.
Something like the following would work

```py
>>> mid = len(a)//2
>>> n = 4
>>> a[mid - n//2: mid + n//2]
['b', 'c', 'd', 'e']
```

From our [sanity check](sanity-check), `mid + n//2 - (mid - n//2)` does
equal `n` if `n` is even (we could find a similar expression for `n` odd, but
for now let us assume `n` is even).

However, let's look at what happens when `n` is larger than the size of `a`:

```py
>>> n = 8
>>> a[mid - n//2: mid + n//2]
['g']
```

This is mostly likely not what we would want. Depending on our use-case, we
would most likely want either an error or the full list `['a', 'b', 'c', 'd',
'e', 'f', 'g']`.

What happened here? Let's look at the slice values:

```py
>>> mid - n//2
-1
>>> mid + n//2
7
```

The `stop` slice value is out of bounds for `a`, but this just causes it
to [clip](clipping) to the end.

But `start` contains a subtraction, which causes it to become negative. Rather
than clipping to the start, it indexes from the end of `a`, producing the
slice `a[-1:7]`. This picks the elements from the last element (`'g'`) up to
but not including the 7th element (0-based). Index `7` is out of bounds for
`a`, so this picks all elements including and after `'g'`, which in this case
is just `['g']`.

Unfortunately, the "correct" fix here depends on the desired behavior for each
individual slice. In some cases, the "slice from the end" behavior of negative
values is in fact what is desired. In others, you might prefer an error, so
should add a value check or assertion. In others, you might want clipping, in
which case you could modify the expression to always be nonnegative. For
example, instead of using `mid - n//2`, we could use `max(mid - n//2, 0)`.

```py
>>> n = 8
>>> a[max(mid - n//2, 0): mid + n//2]
['a', 'b', 'c', 'd', 'e', 'f', 'g']
```

(clipping)=
### Clipping

Slices can never give an out-of-bounds `IndexError`. This is different from
[integer indices](integer-indices) which require the index to be in bounds.
**If `start` or `stop` index before the beginning or after the end of the
`a`, they will clip to the bounds of `a`**:

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[-100:100]
['a', 'b', 'c', 'd', 'e', 'f', 'g']
```

Furthermore, if the `start` is on or after the `stop`, the slice will be
empty.

```py
>>> a[3:3]
[]
>>> a[5:2]
[]
```

For NumPy arrays, a consequence of this is that a slice will always keep the
axis being sliced, even if the size of the resulting axis is 0 or 1.

```py
>>> import numpy as np
>>> arr = np.array([[1, 2], [3, 4]])
>>> arr.shape
(2, 2)
>>> arr[0].shape # Integer index removes the first dimension
(2,)
>>> arr[0:1].shape # Slice preserves the first dimension
(1, 2)
>>> arr[0:0].shape # Slice preserves the first dimension as an empty dimension
(0, 2)
```

An important consequence of the clipping behavior of slices is that you cannot
rely on runtime checks for out-of-bounds slices. See the [example
above](negative-indices-example). Another consequence is that you can never
rely on the length of a slice being `stop - start` (for `step = 1` and
`start`, `stop` nonnegative). This is rather the *maximum* length of the
slice. It could end up slicing something smaller. For example, an empty list
will always slice to an empty list. ndindex can help in calculations here:
`len(ndindex.Slice(...))` can be used to compute the *maximum* length of a
slice. If the shape of the input is known,
`len(ndindex.Slice(...).reduce(shape))` will compute the true length of the
slice (see {meth}`ndindex.Slice.__len__` and {meth}`ndindex.Slice.reduce`). Of
course, if you already have a NumPy array, you can just slice the array and
check the shape (slicing a NumPy array always produces a view on the array, so
it is a very inexpensive operation).

(steps)=
### Steps

Thus far, we have only considered slices with the default step size of 1. When
the step is greater than 1, the slice picks every `step` element contained in
the bounds of `start` and `stop`.

**The proper way to think about `step` is that the slice starts at `start` and
successively adds `step` until it reaches an index that is at or past the
`stop`, and then stops without including that index.**

The important thing to remember about the `step` is that it being non-1 does
not change the fundamental [rules](rules) of slices that we have learned so
far. `start` and `stop` still use [0-based indexing](0-based). The `stop` is
[never included](half-open) in the slice. [Negative](negative-indices) `start`
and `stop` index from the end of the list. Out-of-bounds `start` and `stop`
still [clip](clipping) to the beginning or end of the list.

Let us consider an example where the step size is `3`.

<div style="text-align:center">
<code style="font-size: 16pt;">a[0:6:3] == ['a', 'd']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
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
    & \color{red}{+3\ (\geq \text{stop})}
\end{array}
\end{aligned}
$$
</div>

```py
>>> a[0:6:3]
['a', 'd']
```

Note that the `start` index, `0`, is included, but the `stop` index, `6`
(corresponding to `'g'`), is *not* included, even though it is a multiple of
`3` away from the start. This is because the `stop` is [never
included](half-open).

It can be tempting to think about the `step` in terms of modular arithmetic.
In fact, it is often the case in practice that you require a `step` greater
than 1 because you are dealing with modular arithmetic in some way. However,
this requires care.

Indeed, we can note that resulting indices `0`, `3` of the above slice
`a[0:6:3]` are all multiples of 3. This is because the `start` index, `0`, is
a multiple of 3. If we instead choose a start index that is $1 \pmod{3}$ then
all the indices would also be $1 \pmod{3}$.

<div style="text-align:center">
<code style="font-size: 16pt;">a[1:6:3] == ['b', 'e']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r c c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    & \color{blue}{\enclose{circle}{1}}
    & \color{red}{2\phantom{,}}
    & \color{red}{3\phantom{,}}
    & \color{blue}{\enclose{circle}{4}}
    & \color{red}{5\phantom{,}}
    & \color{red}{\underline{6}\phantom{,}}
    & \color{red}{\enclose{circle}{\phantom{7}}}\\
    &
    & \color{blue}{\text{start}}
    &
    & \rightarrow
    & \color{blue}{+3}
    &
    & \rightarrow
    & \color{red}{+3\ (\geq \text{stop})}
\end{array}
\end{aligned}
$$
</div>

```py
>>> a[1:6:3]
['b', 'e']
```

However, be careful as this rule is *only* true for nonnegative `start`. If
`start` is negative, the value of `start (mod step)` has no
bearing on the indices chosen for the slice:

```py
>>> list(range(21))[-15::3]
[6, 9, 12, 15, 18]
>>> list(range(22))[-15::3]
[7, 10, 13, 16, 19]
```

In the first case, `-15` is divisible by 3 and all the indices chosen by the
slice `-15::3` were also divisible by 3 (remember that indices and values
are the same for simple ranges). But this is only because the length of the
list, `21`, also happened to be a multiple of 3. In the second example it is
`22` and the resulting indices are not multiples of `3`.

Another thing to be aware of is that if the start is [clipped](clipping), the
clipping occurs *before* the step. That is, if the `start` is less than
`len(a)`, it is the same as `start = 0` regardless of the `step`.

```py
>>> a[-100::2]
['a', 'c', 'e', 'g']
>>> a[-101::2]
['a', 'c', 'e', 'g']
```

If you need to think about steps in terms of modular arithmetic,
[`ndindex.Slice()`](slice-api) can be used to perform various slice
calculations so that you don't have to come up with modulo formulas yourself.
If you try to write such formulas yourself, chances are you will get them
wrong, as it is easy to fail to properly account for [negative vs. nonnegative
indices](negative-indices), [clipping](clipping), and [negative
steps](negative-steps). As noted before, any correct "formula" regarding
slices will necessarily have many piecewise conditions.

(negative-steps)=
### Negative Steps

Recall what we said [above](steps):

**The proper way to think about `step` is that the slice starts at `start` and
successively adds `step` until it reaches an index that is at or past the
`stop`, and then stops without including that index.**

The key thing to remember with negative `step` is that this rule still
applies. That is, the index starts at `start` then adds the `step` (which
makes the index smaller), and stops when it is at or past the `stop`. Note the
phrase "at or past". If the `step` is positive this means "greater than or
equal to", but if the `step` is negative this means "less than or equal to".

Think of a slice as starting at the `start` and sliding along the list,
jumping along by `step`, and spitting out elements. Once you see that you are
at or have gone past the `stop` in the direction you are going (left for
negative `step` and right for positive `step`), you stop.

It's worth pointing out that unlike all other slices we have seen so far, a
negative `step` reverses the order that the elements are returned relative to
the original list. In fact, one of the most common uses of a negative `step` is
`a[::-1]`, which reverses the list:

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[::-1]
['g', 'f', 'e', 'd', 'c', 'b', 'a']
```

It is tempting therefore to think of a negative `step` as a "reversing"
operation. However, this is a bad way of thinking about negative steps. The
reason is that `a[i:j:-1]` is *not* equivalent to `reversed(a[j:i:1])`. The
reason is basically the same as was described in [wrong rule 1](wrong-rule-1)
above. The issue is that for `a[start:stop:step]`, `stop` is *always* what is
[not included](half-open). Which means if we swap `i` and `j`, we go from "`j`
is not included" to "`i` is not included", producing a wrong result. For
example, as [before](wrong-rule-1):

```py
>>> a[5:3:-1]
['f', 'e']
>>> list(reversed(a[3:5:1])) # This is not the same thing
['e', 'd']
```

In the first case, index `3` is not included. In the second case, index `5` is
not included.

Worse, this way of thinking may even lead one to imagine the completely wrong
idea that `a[i:j:-1]` is the same as `reversed(a)[j:i]`:

```py
>>> list(reversed(a))[3:5]
['d', 'c']
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
moving across the list by `step` until reaching `stop`, which is not included.

Negative steps can of course be less than -1 as well, with similar behavior to
steps greater than 1, again, keeping in mind that the `stop` is not included.

```py
>>> a[6:0:-3]
['g', 'd']
```

<div style="text-align:center">
<code style="font-size: 16pt;">a[6:0:-3] == ['g', 'd']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, & \mathtt{\textsf{'}b\textsf{'}}, & \mathtt{\textsf{'}c\textsf{'}}, & \mathtt{\textsf{'}d\textsf{'}}, & \mathtt{\textsf{'}e\textsf{'}}, & \mathtt{\textsf{'}f\textsf{'}}, & \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{red}{\enclose{circle}{0}\phantom{,}}
    & \color{red}{1\phantom{,}}
    & \color{red}{2\phantom{,}}
    & \color{blue}{\enclose{circle}{3}}
    & \color{red}{4\phantom{,}}
    & \color{red}{5\phantom{,}}
    & \color{blue}{\enclose{circle}{6}}\\
    & \color{red}{-3}\phantom{\mathtt{\textsf{'},}}
    & \leftarrow
    &
    & \color{blue}{-3}\phantom{\mathtt{\textsf{'},}}
    & \leftarrow
    &
    & \color{blue}{\text{start}}\\
    &  \color{red}{(\leq \text{stop})}
\end{array}
\end{aligned}
$$
</div>

The `step` can never be equal to 0. This unconditionally produces an error:

```py
>>> a[::0]
Traceback (most recent call last):
...
ValueError: slice step cannot be zero
```

(omitted)=
### Omitted Entries

The final point of confusion is omitted entries.[^omitted-none-footnote]

[^omitted-none-footnote]: `start`, `stop`, or `step` may also be `None`, which
is syntactically equivalent to them being omitted. That is to say, `a[::]` is
a syntactic shorthand for `a[None:None:None]`. It is rare to see `None` in a
slice. This is only relevant for code that consumes slices, such as a
`__getitem__` method on an object. The `slice()` object corresponding to
`a[::]` is `slice(None, None, None)`. [`ndindex.Slice()`](slice-api) also uses
`None` to indicate omitted entries in the same way.

**The best way to think about omitted entries is just like that, as omitted
entries.** That is, for a slice like `a[:i]`, think of it as the `start` being
omitted, and `stop` equal to `i`. Conversely, `a[i:]` has the `start` as `i`
and the `stop` omitted. The wrong way to think about these is as a colon being
before or after the index `i`. Thinking about it this way will only lead to
confusion, because you won't be thinking about `start` and `stop`, but rather
trying to remember some rule based on where a colon is. But the colons in a
slice are not indicators, they are separators.

As to the semantic meaning of omitted entries, the easiest one is the `step`.
**If the `step` is omitted, it always defaults to `1`.** If the `step` is
omitted the second colon before the `step` can also be omitted. That is to
say, the following are completely equivalent:

```py
a[i:j:1]
a[i:j:]
a[i:j]
```

<!-- TODO: Better wording for this rule? -->

**For the `start` and `stop`, the rule is that being omitted extends the slice
all the way to the beginning or end of `a` in the direction being sliced.** If
the `step` is positive, this means `start` extends to the beginning of `a` and
`stop` extends to the end. If `step` is negative, it is reversed: `start`
extends to the end of `a` and `stop` extends to the beginning.

<div style="text-align:center">
<code style="font-size: 16pt;">a[:3] == a[:3:1] == ['a', 'b', 'c']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r c c c c c c c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, && \mathtt{\textsf{'}b\textsf{'}}, && \mathtt{\textsf{'}c\textsf{'}}, && \mathtt{\textsf{'}d\textsf{'}}, && \mathtt{\textsf{'}e\textsf{'}}, && \mathtt{\textsf{'}f\textsf{'}}, && \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{blue}{\enclose{circle}{0}}
    &
    & \color{blue}{\enclose{circle}{1}}
    &
    & \color{blue}{\enclose{circle}{2}}
    &
    & \color{red}{\enclose{circle}{3}}
    &
    & \color{red}{4\phantom{,}}
    &
    & \color{red}{5\phantom{,}}
    &
    & \color{red}{6\phantom{,}}\\
    \color{blue}{\text{start}}
    & \color{blue}{\text{(beginning)}}
    & \rightarrow
    & \color{blue}{+1}
    & \rightarrow
    & \color{blue}{+1}
    & \rightarrow
    & \color{red}{\text{stop}}
    &
    & \phantom{\rightarrow}
    &
    & \phantom{\rightarrow}
    &
    & \phantom{\rightarrow}
\end{array}
\end{aligned}
$$
</div>

<div style="text-align:center">
<code style="font-size: 16pt;">a[3:] == a[3::1] == ['d', 'e', 'f', 'g']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r c c c c c c c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, && \mathtt{\textsf{'}b\textsf{'}}, && \mathtt{\textsf{'}c\textsf{'}}, && \mathtt{\textsf{'}d\textsf{'}}, && \mathtt{\textsf{'}e\textsf{'}}, && \mathtt{\textsf{'}f\textsf{'}}, && \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    &
    & \color{red}{1\phantom{,}}
    &
    & \color{red}{2\phantom{,}}
    &
    & \color{blue}{\enclose{circle}{3}}
    &
    & \color{blue}{\enclose{circle}{4}}
    &
    & \color{blue}{\enclose{circle}{5}}
    &
    & \color{blue}{\enclose{circle}{6}\phantom{,}}\\
    &
    & \phantom{\rightarrow}
    &
    & \phantom{\rightarrow}
    &
    & \phantom{\rightarrow}
    & \color{blue}{\text{start}}
    & \rightarrow
    & \color{blue}{+1}
    & \rightarrow
    & \color{blue}{+1}
    & \rightarrow
    & \color{blue}{\text{stop}}
    & \color{blue}{\text{(end)}}
\end{array}
\end{aligned}
$$
</div>

<div style="text-align:center">
<code style="font-size: 16pt;">a[:3:-1] == ['g', 'f', 'e']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r c c c c c c c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, && \mathtt{\textsf{'}b\textsf{'}}, && \mathtt{\textsf{'}c\textsf{'}}, && \mathtt{\textsf{'}d\textsf{'}}, && \mathtt{\textsf{'}e\textsf{'}}, && \mathtt{\textsf{'}f\textsf{'}}, && \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{red}{0\phantom{,}}
    &
    & \color{red}{1\phantom{,}}
    &
    & \color{red}{2\phantom{,}}
    &
    & \color{red}{\enclose{circle}{3}}
    &
    & \color{blue}{\enclose{circle}{4}}
    &
    & \color{blue}{\enclose{circle}{5}}
    &
    & \color{blue}{\enclose{circle}{6}\phantom{,}}\\
    &
    & \phantom{\leftarrow}
    &
    & \phantom{\leftarrow}
    &
    & \phantom{\leftarrow}
    & \color{red}{\text{stop}}
    & \leftarrow
    & \color{blue}{-1}
    & \leftarrow
    & \color{blue}{-1}
    & \leftarrow
    & \color{blue}{\text{start}}
    & \color{blue}{\text{(end)}}
\end{array}
\end{aligned}
$$
</div>

<div style="text-align:center">
<code style="font-size: 16pt;">a[3::-1] == ['d', 'c', 'b', 'a']</code>
$$
\require{enclose}
\begin{aligned}
\begin{array}{r r c c c c c c c c c c c c l}
a = & [\mathtt{\textsf{'}a\textsf{'}}, && \mathtt{\textsf{'}b\textsf{'}}, && \mathtt{\textsf{'}c\textsf{'}}, && \mathtt{\textsf{'}d\textsf{'}}, && \mathtt{\textsf{'}e\textsf{'}}, && \mathtt{\textsf{'}f\textsf{'}}, && \mathtt{\textsf{'}g\textsf{'}}]\\
\color{red}{\text{index}}
    & \color{blue}{\enclose{circle}{0}}
    &
    & \color{blue}{\enclose{circle}{1}}
    &
    & \color{blue}{\enclose{circle}{2}}
    &
    & \color{blue}{\enclose{circle}{3}}
    &
    & \color{red}{4\phantom{,}}
    &
    & \color{red}{5\phantom{,}}
    &
    & \color{red}{6\phantom{,}}\\
    \color{blue}{\text{stop}}
    & \color{blue}{\text{(beginning)}}
    & \leftarrow
    & \color{blue}{-1}
    & \leftarrow
    & \color{blue}{-1}
    & \leftarrow
    & \color{blue}{\text{start}}
    &
    & \phantom{\leftarrow}
    &
    & \phantom{\leftarrow}
    &
    & \phantom{\leftarrow}
\end{array}
\end{aligned}
$$
</div>

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[:3] # From the beginning to index 3 (but not including index 3)
['a', 'b', 'c']
>>> a[3:] # From index 3 to the end
['d', 'e', 'f', 'g']
>>> a[:3:-1] # From the end to index 3 (but not including index 3), reversed
['g', 'f', 'e']
>>> a[3::-1] # From index 3 to the beginning, reversed
['d', 'c', 'b', 'a']
```

## Soapbox

While this guide is opinionated about the right and wrong ways to think about
slices in Python, I have tried to stay neutral regarding the merits of the
rules themselves. But I want to take a moment to give my views on them. I have
worked with slice objects quite a bit in building ndindex, as well as just
general usage with Python and NumPy.

Python's slice syntax is, without a doubt, extremely expressive, and has a
straightforward and simple syntax. However, simply put, the semantic rules for
slices are completely bonkers. They lend themselves to several invalid
interpretations, which I have outlined above, which seem valid at first glance
but fall apart in corner cases. The "correct" ways to think about slices are
very particular. I have tried to [outline](rules) them carefully, but one gets
the impression that unless one works with slices regularly, it will be hard to
remember the "right" ways and not fallback to thinking about the "wrong" ways,
or, as most Python programmers probably do, simply "guessing and checking".

Furthermore, the discontinuous nature of the `start` and `stop` parameters not
only makes it hard to remember how slices work, but it makes it *extremely*
hard to write slice arithmetic. The arithmetic is already hard enough due to
the modular nature of `step`, but the discontinuous aspect of `start` and
`stop` increases this tenfold. If you are unconvinced of this, take a look at
the [source
code](https://github.com/Quansight/ndindex/blob/master/ndindex/slice.py) for
`ndindex.Slice()`. You will see lots of nested `if` blocks.[^source-footnote]
This is because slices have *fundamentally* different definitions if the
`start` or `stop` are `None`, negative, or nonnegative. Furthermore, `None` is
not an integer, so one must always be careful to either check for it first or
to be certain that it cannot happen, before performing any arithmetical
operation or numerical comparison. Under each `if` block you will see some
formula or other. Many of these formulas were difficult to come up with. In
many cases they are asymmetrical in surprising ways. It is only through the
rigorous [testing](testing) that ndindex uses that I can have confidence the
formulas are correct for all corner cases.

[^source-footnote]: To be sure, I make no claims that the source of any
function in ndindex cannot be simplified. In writing ndindex, I have primarily
focused on making the logic correct, and less on making it elegant. I welcome
any pull requests that simplifies the logic of a function. The extensive
[testing](testing) should ensure that any rewritten function remains correct.

It is my opinion that Python's slicing semantics could be just as expressive,
but much less confusing and difficult to work with, both for end-users of
slices and people writing slice arithmetic (a typical user of ndindex). The
changes I would make to improve the semantics would be

1. Remove the special meaning of negative numbers.
2. Use 1-based indexing instead of 0-based indexing.
3. Make a slice always include both the start and the stop.

**Negative numbers.** The special meaning of negative numbers, to index from
the end of the list, is by far the biggest problem with Python's slice
semantics. It is a fundamental discontinuity in the definition of an index.
This makes it completely impossible to write a formula for almost anything
relating to slices that will not end up having branching `if` conditions. But
the problem isn't just for code that manipulates slices. The [example
above](negative-indices-example) shows how negative indices can easily lead to
bugs in end-user code. Effectively, any time you have a slice `a[i:j]`, if `i`
and `j` are nontrivial expressions, they must be checked to ensure they do not
go negative. If they can be both negative and nonnegative, it is virtually
never the case that the slice will give you what you want in both cases. This
is because the discontinuity inherent in the definition of [negative
indexing](negative-indices) disagrees with the concept of
[clipping](clipping). `a[i:j]` will slice "as far as it can" if `j` is "too
big" (greater than `len(a)`), but it does something completely different if
`i` is "too small" as soon as "too small" means "negative". Clipping is a good
idea. It tends to lead to behavior that gives what you would want slices that
go out of bounds.

Negative indexing is, strictly speaking, a syntactic sugar only.
Slicing/indexing from the end of a list can always be done in terms of the
length of the list. `a[-x]` is the same as `a[len(a)-x]` (when using 0-based
indexing), but the problem is that it is tedious to write `a` twice, and `a`
may in fact be a larger expression, so writing `a[len(a)-x]` would require
assigning it to a variable. It also becomes more complicated when `a` is a
NumPy array and the slice appears as part of a larger multidimensional (tuple)
index. However, I think it would be possible to introduce a special syntax to
mean "reversed" or "from the end the list" indexing, and leave negative
numbers to simply extend beyond the left side of a list with clipping. For
example, in [Julia](https://julialang.org/), one can use `a[end]` to index the
last element of an array (Julia also uses 1-based indexing). Since this is a
moot point for Python---I don't expect Python's indexing semantics to change;
they are already baked into the language---I won't suggest any syntax. Perhaps
this can inspire people writing new languages or DSLs to come up with better
semantics backed by good syntax (again, I think Python slicing has good
*syntax*. I only take issue with some of its *semantics*.).

**0-based vs. 1-based indexing.** The second point, on using 1-based indexing
instead of 0-based indexing, will likely be the most controversial. For many
people reading this, the notion that 0-based indexing is superior has been
preached as irreproachable gospel. I encourage you to open your mind and try
to unlearn what you have been taught and take a fresh view of the matter (or
don't. These are just my opinions after all, and none of it changes the fact
that Python is what it is and isn't going to change).

0-based indexing certainly has its uses. In C, where an index is literally a
syntactic macro for adding two pointers, 0-based indexing makes sense, since
`a[i]` literally means `*(a + i)` under those semantics. However, for higher
level languages such as Python, people think of indexing as pointing to
specific numbered elements of a collection, not as pointer arithmetic. Every
human being is taught from an early age to count from 1. If you show someone
the list "a, b, c", they will tell you that "a" is the 1st, "b" is the 2nd,
and "c" is the 3rd. [Sentences](fourth-sentence) in the above guide like
"`a[3]` would pick the fourth element of `a`" sound very off, even for those
of us used to 0-based indexing. 0-based indexing requires a shift in thinking
from the way that you have been taught to count from early childhood. Counting
is a very fundamental thing for any human, but especially so for a programmer.
Forcing someone to learn a new way to do such a foundational thing is a huge
cognitive burden, and so it shouldn't be done without a very good reason. In a
language like C, one can argue there is a good reason, just as one can argue
that it is beneficial to learn new base number systems like base-2 and base-16
when doing certain kinds of programming.

But for Python, what truly is the benefit of counting starting at 0? The main
benefit is that the implementation is easier, because Python is itself written
in C, which uses 0-based indexing, so Python does not need to handle shifting
in the translation. But this has never been a valid argument for Python
semantics. The whole point of Python is to provide higher level semantics than
C, and leave those hard details of translating them to the interpreter and
library code. In fact, Python's slices themselves are much more complicated
than what is available in C, and the interpreter code to handle them is more
than just a trivial translation to C. Adding shifts to this translation code
would not be much additional complexity.

Even experienced programmers of languages like Python that use 0-based
indexing must occasionally stop themselves from writing something like `a[3]`
instead of `a[2]` to get the third element of `a`. It is very difficult to
"unlearn" 1-based counting," which was not only the first way that you learned
to count, but is also the way that you and everyone else around you continues
to count outside of programming contexts.

When you teach a child how to count things, you teach them to enumerate the
items starting at 1 ("1, 2, 3, ..."). The number that is enumerated for the
final object is equal to the number of items (the final ordinal is equal to
the cardinal). This only works if you start at 1. If the child instead starts
at 0 ("0, 1, 2, ...") the final ordinal (the last number spoken aloud) would
not match the cardinal (the number of items). The distinction between ordinals
and cardinals is not something most people think about often, because the
convention of counting starting at 1 makes it so that they are equal. But as
programmers in a language that rejects this elegant convention, we are forced
to think about such philosophical distinctions just to solve whatever problem
we are trying to solve.

In most instances (outside of programming) where a reckoning starts at 0
instead of 1, it is because it is measuring a distance. The distance from your
house to the nearest pub may be "2 miles", but the distance from your house to
itself is "0 miles". On the other hand, when counting or enumerating
individual objects, counting always starts at 1. The notion of a "zeroth"
object doesn't make sense when counting say apples, because you are counting
the apples themselves, not some quantity relating them. It doesn't make sense
to start at 0 with an "un-apple" because you are only counting apples, not
non-apples.

So the question then becomes, should indexing work like a measurement of
distance, which would naturally start at 0, or like an enumeration of distinct
terms, which would naturally start at 0. If we think of an index as a pointer
offset, as C does, then it is indeed a measurement of a distance. But if we
instead think of an indexable list as a discrete ordered collection of items,
then the notion of a measurement of distance is harder to justify. But
enumeration is a natural concept for any ordered discrete collection.

What are the benefits of 0-based indexing?

- It makes translation to lower level code (like C or machine code) easier.
  But as I already argued, this is not a valid argument for Python, which aims
  to be high-level and abstract away translation complexities that make coding
  more difficult. The translation that necessarily takes place in the
  interpreter itself can afford this complexity if it means making the
  language itself simpler.
- It makes translation of code written in other languages that use 0-based
  indexing simpler. If Python used 1-based indexing, then to translate a C
  algorithm to Python, for instance, one would have to adapt all the places
  that use indexing, which would be a bug-prone task. But Python's primary
  mantra is having syntax and semantics that make code easy to read and easy
  to write. Being similar to other existing languages is second to this, and
  should not take priority when it conflicts with it. Translation of code from
  other languages to Python does happen, but it is much rarer than novel code
  written in Python. Furthermore, automated tooling could be used to avoid
  translation bugs. Such tooling would help avoid other translation bugs
  unrelated to indexing as well.
- It works nicely with half-open semantics. It is true that half-open
  semantics and 0-based indexing, while technically distinct, are virtually
  always implemented together because they play so nicely with each other.
  However, as I argue below, half-open semantics are just as absurd as 0-based
  indexing, and abandoning both for the more standard closed-closed/1-based
  semantics is very reasonable.

To me, the ideal indexing system defaults to 1-based, but allows starting at
any index. That way, if you are dealing with a use-case where 0-based indexing
really does make more sense, you can easily use it. Indices should also be
able to start at any other number, including negative numbers (which is
another reason to remove the special meaning of negative indices). An example
of a use-case where 0-based indexing truly is more natural than 1-based
indexing is polynomials. Say we have a polynomial $a_0 + a_1x + a_2x^2 +
\cdots$. Then we can represent the coefficients $a_0, a_1, a_2, \ldots$ in a
list `[a0, a1, a2, ...]`. Since a polynomial naturally has a 0th coefficient,
it makes sense to index the list starting at 0 (and one must still be careful
about off-by-one errors; a degree-$n$ polynomial has $n+1$ coefficients).

If this seems like absurd idea, note that this is how Fortran works (see
<https://www.fortran90.org/src/faq.html#what-is-the-most-natural-starting-index-for-numbering>).
In Fortran, arrays index starting at 1 by default, but any integer can be used
as a starting index. Fortran predates Python by many decades, but is still in
use today, particularly in scientific applications, and many Python libraries
themselves such as SciPy are backed by Fortran code. Many other popular
programming languages use 1-based indexing, such as Julia, MATLAB,
Mathematica, R, Lua, and
[others](https://en.wikipedia.org/wiki/Comparison_of_programming_languages_(array)#Array_system_cross-reference_list).
In fact, a majority of the popular programming languages that use 1-based
indexing are languages that are primarily used for scientific applications.
Scientific applications tend to make much heavier use of arrays than most
other programming tasks, and hence a heavy use of indexing.

**Half-open semantics.** Finally, the idea of half-open semantics, where the
`stop` value of a slice is never included, is bad, for many of the same
reasons that 0-based indexing is bad. In most contexts outside of programming,
including virtually all mathematical contexts, when one sees a range of
values, it is implicitly assumed that both endpoints are included in the
range. For example, if you see a phrase like "ages 7 to 12", "the letters A to
Z", or "sum of the numbers from 1 to 10", without any further qualification
you assume that both endpoints are included in the range. Half-open semantics
also break down when considering non-numeric quantities. For example, one
cannot represent the set of letters "from A to Z" except by including both
endpoints, as there is no letter after Z to not include.

It is simply more natural to think about a range as including both endpoints.
Half-open semantics are often tied to 0-based indexing, since it is a
convenient way to allow the range 0--N to contain N values, by not including
N.[^python-history-footnote] I see this as taking a bad decision (0-based
indexing) and putting a bad bandaid on it that makes it worse. But certainly
this argument goes away for 1-based indexing. The range 1--N contains N values
exactly when N *is* included in the range.

[^python-history-footnote]: In fact, the original reason that Python uses
0-based indexing is that Guido preferred the half-open semantics, which only
work out well when combined with 0-based indexing
([reference](https://web.archive.org/web/20190321101606/https://plus.google.com/115212051037621986145/posts/YTUxbXYZyfi)).

You might argue that there are instances in everyday life where half-open as
well as 0-based semantics are used. For example, in the West, the reckoning of
a person's age is typically done in a way that matches half-open 0-based
indexing semantics. If has been less than 1 year since a person's birthdate,
you might say they are "zero years old" (although typically you use a smaller
unit of measure such as months to avoid this). And if tomorrow is my 30th
birthday, then today I will say, "I am 29 years old", even though I am
actually 29.99 years old (I may continue to say "I am 29 years old" tomorrow,
but at least today no one would accuse me of lying). This matches the
"half-open" semantics used by slices. The end date of an age, the birthday, is
not accounted for until it has passed. This example shows that half-open
semantics do indeed go nicely with 0-based counting, and it's indeed typically
good to use one when using the other. But age is a distance. It is the
distance in time since a person's birthdate. So 0-based indexing makes sense
for it. Half-open semantics play nicely with age not just because it lets us
lie to ourselves about being younger than we really are, but because age is a
continuous quantity which is reckoned by integer values for convenience. Since
people rarely concern themselves with fractional ages, they must increment an
age counter at some point, and doing so on a birthday, which leads to a
"half-open" semantic, makes sense. But a collection of items like a list,
array, or string in Python usually does not represent a continuous quantity
which is discretized, but rather a quantity that is naturally discrete. So
while half-open 0-indexed semantics are perfectly reasonable for human ages,
the same argument doesn't make sense for collections in Python.

When it comes to indexing, half-open semantics are problematic for a few
reasons:

- A commonly touted benefit of half-open slicing semantics is that you can
  "glue" half-open intervals together. For example, `a[0:N] + a[N:M]` is the
  same as `a[0:M]`. But `a[1:N] + a[N+1:M]` is just as clear. People are
  perfectly used to adding 1 to get to the next term in a sequence, and it is
  more natural to see `[1:N]` and `[N+1:M]` as being non-overlapping if they
  do not share endpoint values. Ranges that include both endpoints are
  standard in both mathematics and everyday language. $\sum_{i=1}^n$ means a
  summation from $1$ to $n$ inclusive. Formulas like $\sum_{i=1}^n a_i =
  \sum_{i=1}^k a_i + \sum_{i=k+1}^n a_i$ are natural to anyone who has studied
  enough mathematics. If you were to rewrite the sentences "the first $n$
  numbers are $1\ldots n$; the next $n$ numbers are $n+1\ldots 2n$", or "'the
  70s' refers to the years 1970--1979" using half-open semantics, anyone would
  tell you they were phrased wrong.

- Another benefit of half-open intervals is that they allow the range `a[i:j]`
  to contain $j - i$ elements (assuming $0 \leq i \leq j$ and `a` is large
  enough). I tout this myself in the guide above, since it is a useful [sanity
  check](sanity-check). However, as useful as it is, it isn't worth the more
  general confusion caused by half-open semantics. I contend people are
  perfectly used to the usual [fencepost](fencepost) offset that a range
  $i\ldots j$ contains $j - i + 1$ numbers. Half-open semantics replace this
  fencepost error with more subtle ones, which arise from forgetting that the
  range doesn't include the endpoint, unlike most natural ranges that occur in
  day-to-day life. See [wrong rule 3](wrong-rule-3) above for an example of how
  half-open semantics can lead to subtle fencepost errors.

  It is true that including both endpoints in range can lead to [fencepost
  errors](fencepost). But the fencepost problem is fundamentally unavoidable. A
  100 foot fence truly has one more fencepost than fence lengths. The best way
  to deal with the fencepost problem is not to try to change the way we count
  fenceposts, so that somehow 11 fenceposts is really only
  10.[^fencepost-footnote] It is rather to reuse the most natural and intuitive
  way of thinking about the problem, which occurs both in programming and
  non-programming contexts, which is that certain quantities, like the number of
  elements in a range $1\ldots N$ will require an extra "+ 1" to be correct.

[^fencepost-footnote]: [This blog
 post](https://betterexplained.com/articles/learning-how-to-count-avoiding-the-fencepost-problem/)
 has a nice writeup of *why* the fencepost problem exists. It's related to the
 difference between measurement and enumeration that I touched on earlier.

- Half-open semantics become particularly confusing when the step is negative.
  This is because one must remember that the end that is not included in the
  half-open interval is the second index in the slice, *not* the larger index
  (see wrong rules [1](wrong-rule-1) and [3](wrong-rule-3) above). Were both
  endpoints included, this confusion would be impossible, because positive and
  negative steps would be symmetric in this regard.


In general, half-open semantics are naively superior because they have some
properties that appear to be nice (easy unions, no +1s in length formulas).
But the "niceness" of these properties ignores the fact that most people are
already used to closed-closed intervals from mathematics and from everyday
life, and so are used to accounting for them already. So while these
properties are nice, they also break the natural intuition of how ranges work.
Half-open semantics are also closely tied to 0-based indexing, which as I
argued above, is itself problematic for many of the same reasons.

Again, there is no way Python itself can change any of these things at this
point. It would be way too big of a change to the language, far bigger than
any change that was made as part of Python 3 (and the Python developers have
already stated that they will never do a big breaking change like Python 3
again). But I hope I can inspire new languages and DSLs that include slicing
semantics to be written in clearer ways. And I also hope that I can break some
of the cognitive dissonance that leads people to believe that the Python
slicing rules are superior, despite the endless confusion that they provide.
Finally, I believe that simply understanding that Python has made these
decisions, whether you agree with them or not, will help you to remember the
slicing [rules](rules), and that's my true goal here.

## Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
