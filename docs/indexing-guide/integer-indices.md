# Integer Indices

The simplest possible index type is an integer index, that is `a[i]` where `i`
is an integer like `0`, `3`, or `-2`.

Integer indexing operates on the familiar Python data types `list`, `tuple`,
and `str`, as well as NumPy arrays.

(prototype-example)=
Let's use as an example this prototype list:

<div class="slice-diagram">
  <table>
    <tr>
      <td><pre>a</pre></td>
      <td><pre>=</pre></td>
      <td><pre>['a',</pre></td>
      <td><pre>'b',</pre></td>
      <td><pre>'c',</pre></td>
      <td><pre>'d',</pre></td>
      <td><pre>'e',</pre></td>
      <td><pre>'f',</pre></td>
      <td><pre>'g']</pre></td>
    </tr>
  </table>
</div>

The list `a` has 7 elements.

The elements of `a` are strings, but the indices and slices on the list `a`
will always use integers. Like [all other index types](what-is-an-index),
**the result of an integer index is never based on the values of the elements,
but rather their position of the elements in the list.**[^dict-footnote]

[^dict-footnote]: If you are looking for something that allows non-integer
indices or that indexes by value, you may want a `dict`.

An integer index picks a single element from the list `a`. For NumPy arrays,
integer indices pick a subarray corresponding to a particular element from a
given axis (and as a result, an integer index always reduces the
dimensionality of an array by one).

(fourth-sentence)=
**The key thing to remember about indexing in Python, both for integer and
slice indexing, is that it is 0-based.** This means that the indices start
counting at 0 (like "0, 1, 2, ..."). This is the case for all *nonnegative*
indices[^nonnegative]. For example, `a[3]` would pick the *fourth* element of
`a`, in this case, `'d'`:

[^nonnegative]: In this guide, "*nonnegative*" means $\geq 0$ and
    "*negative*" means $< 0$.

<div class="slice-diagram">
<code style="font-size: 16pt;">a[3] == 'd'</code>
  <table>
    <tr>
      <td><pre>a</pre></td>
      <td><pre>=</pre></td>
      <td><pre>['a',</pre></td>
      <td><pre> 'b',</pre></td>
      <td><pre> 'c',</pre></td>
      <td><pre> 'd',</pre></td>
      <td><pre> 'e',</pre></td>
      <td><pre> 'f',</pre></td>
      <td><pre> 'g']</pre></td>
    </tr>
    <tr>
      <th style="color:var(--color-slice-diagram-not-selected);">index</th>
      <td></td>
      <td style="color:var(--color-slice-diagram-not-selected);">0</td>
      <td style="color:var(--color-slice-diagram-not-selected);">1</td>
      <td style="color:var(--color-slice-diagram-not-selected);">2</td>
      <td style="color:var(--color-slice-diagram-selected);">3</td>
      <td style="color:var(--color-slice-diagram-not-selected);">4</td>
      <td style="color:var(--color-slice-diagram-not-selected);">5</td>
      <td style="color:var(--color-slice-diagram-not-selected);">6</td>
    </tr>
  </table>
</div>

```py
>>> a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> a[3]
'd'
```

0-based indexing is different from how people typically count things, which is
1-based ("1, 2, 3, ..."). Thinking in terms of 0-based indexing requires some
practice, but doing so is essential to becoming an effective Python
programmer, especially if you are planning to work with arrays.

For *negative* integers, indices index from the end of the list. These indices
are necessarily 1-based (or rather, &minus;1-based), since `0` already refers
to the first element of the list. `-1` chooses the last element, `-2` the
second-to-last, and so on. For example, `a[-3]` picks the *third-to-last*
element of `a`, in this case, `'e'`:


<div class="slice-diagram">
<code style="font-size: 16pt;">a[-3] == 'e'</code>
  <table>
    <tr>
      <td><pre>a</pre></td>
      <td><pre>=</pre></td>
      <td><pre>['a',</pre></td>
      <td><pre> 'b',</pre></td>
      <td><pre> 'c',</pre></td>
      <td><pre> 'd',</pre></td>
      <td><pre> 'e',</pre></td>
      <td><pre> 'f',</pre></td>
      <td><pre> 'g']</pre></td>
    </tr>
    <tr>
      <th style="color:var(--color-slice-diagram-not-selected);">index</th>
      <td></td>
      <td style="color:var(--color-slice-diagram-not-selected);">&minus;7</td>
      <td style="color:var(--color-slice-diagram-not-selected);">&minus;6</td>
      <td style="color:var(--color-slice-diagram-not-selected);">&minus;5</td>
      <td style="color:var(--color-slice-diagram-not-selected);">&minus;4</td>
      <td style="color:var(--color-slice-diagram-selected);">&minus;3</td>
      <td style="color:var(--color-slice-diagram-not-selected);">&minus;2</td>
      <td style="color:var(--color-slice-diagram-not-selected);">&minus;1</td>
    </tr>
  </table>
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
than negative the size of the list (`i >= len(a)` or `i < -len(a)`), then it
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

For NumPy arrays, this applies to the size of the axis being indexed (not the
total size of the array):


```
>>> import numpy as np
>>> a = np.ones((2, 3)) # A has 6 elements but the first axis is size 2
>>> a[2]
Traceback (most recent call last):
...
IndexError: index 2 is out of bounds for axis 0 with size 2
>>> a[-3]
Traceback (most recent call last):
...
IndexError: index -3 is out of bounds for axis 0 with size 2
```

Fortunately, NumPy arrays give a more helpful error message for `IndexError`
than Python does for `list`.

The second important fact about integer indexing is that it reduces the
dimensionality of the container being indexed. For a `list` or `tuple`, this
means that an integer index returns an element of the list, which is in
general a different type than `list` or `tuple`. For instance, above we saw
that indexing `a` with an integer resulted in a `str`, because `a` is a list
that contains strings. This is in contrast with [slices](slices-docs), which
always [return a container type](subarray).

The exception to this rule is when integer indexing a
`str`, the result is also a `str`. This is because there is no `char` class in
Python. A single character is just represented as a string of length 1.

```py
>>> 'abc'[0]
'a'
>>> type('abc'[0])
<class 'str'>
```

For NumPy arrays, an integer index always indexes a single axis of the array.
By default, it indexes the first axis, unless it is part of a larger
[multidimensional index](multidimensional-indices). The resulting array is always an
array with the dimensionality reduced by 1, namely, the axis being indexed is
removed from the resulting shape. This is contrast with [slices](slices-docs), which always
[maintain the dimension being sliced](subarray).

```py
>>> a = np.ones((2, 3, 4))
>>> a.shape
(2, 3, 4)
>>> a[0].shape
(3, 4)
>>> a[-1].shape
(3, 4)
>>> a[..., 0].shape # Index the last axis, see the section on ellipses
(2, 3)
```

One way to think about integer indexing on a NumPy array is to think about the
list-of-lists analogy. An integer index on the first axis `a[i]` picks the
index `i` sub-list at the top level of sub-list nesting, and in general, an
integer index `i` on axis `k` picks the sub-lists of index `i` at the `k`-th
nesting level.[^nesting-level] See the [](what-is-an-array) section on the
[](multidimensional-indices) page for more details on this analogy. For example, if
`l` is a nested list of lists

[^nesting-level]: Thinking about the `k`-th level of nesting gets confusing
    once you start thinking about whether or not `k` is counted with 0-based
    number, and which level counts as which considering that at the outermost
    "level" there is always a single list. List-of-lists is a good analogy for
    thinking about why one might want to use an nd-array in the first place,
    but as you actually use NumPy arrays in practice, you'll find it's much
    better to think about dimensions and axes, not "levels of nesting".

```py
>>> l = [[0, 1], [2, 3]]
```

And `a` is the corresponding array:

```
>>> a = np.array(l)
```

Then `a[0]` is the same thing as `l[0]`, the first sub-list:

```
>>> a[0]
array([0, 1])
>>> l[0]
[0, 1]
```

If we instead index the second axis, like `a[:, 0]`, this is the same as
indexing `0` in each list inside of `l`, like

```
>>> [x[0] for x in l]
[0, 2]
>>> a[:, 0]
array([0, 2])
```

## Footnotes
