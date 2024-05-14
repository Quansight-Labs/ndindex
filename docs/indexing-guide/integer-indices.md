# Integer Indices

The simplest possible index type is an integer index, that is, `a[i]` where `i`
is an integer like `0`, `3`, or `-2`.

Integer indexing operates on the familiar Python data types `list`, `tuple`,
and `str`, as well as NumPy arrays.

(prototype-example)=
Let us consider the following prototype list as an example:

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
will always use integers. As with [all other index types](intro.md), **the
result of an integer index is never based on the values of the elements; it is
based instead on their positions in the list.**[^dict-footnote]

[^dict-footnote]: If you are looking for something that allows non-integer
indices or that indexes by value, you may want a `dict`.

An integer index selects a single element from the list `a`.

> **The key thing to remember about indexing in Python, both for integer and
  slice indexing, is that it is 0-based.**

(fourth-sentence)=
This means that indices start at 0 ("0, 1, 2, ..."). For example,
`a[3]` selects the *fourth* element of `a`, in this case, `'d'`:

<div class="slice-diagram">
<code style="font-size: 16pt;">a[<span class="slice-diagram-slice">3</span>] == 'd'</code>
  <table>
    <tr>
      <td><pre>a</pre></td>
      <td><pre>=</pre></td>
      <td><pre>['a',</pre></td>
      <td><pre> 'b',</pre></td>
      <td><pre> 'c',</pre></td>
      <td class="underline-cell"><pre> 'd',</pre></td>
      <td><pre> 'e',</pre></td>
      <td><pre> 'f',</pre></td>
      <td><pre> 'g']</pre></td>
    </tr>
    <tr>
      <th>index</th>
      <td></td>
      <td class="slice-diagram-not-selected">0</td>
      <td class="slice-diagram-not-selected">1</td>
      <td class="slice-diagram-not-selected">2</td>
      <td class="slice-diagram-selected">3</td>
      <td class="slice-diagram-not-selected">4</td>
      <td class="slice-diagram-not-selected">5</td>
      <td class="slice-diagram-not-selected">6</td>
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
to the first element of the list. `-1` selects the last element, `-2` the
second-to-last, and so on. For example, `a[-3]` selects the *third-to-last*
element of `a`, in this case, `'e'`:


<div class="slice-diagram">
<code style="font-size: 16pt;">a[<span class="slice-diagram-slice">-3</span>] == 'e'</code>
  <table>
    <tr>
      <td><pre>a</pre></td>
      <td><pre>=</pre></td>
      <td><pre>['a',</pre></td>
      <td><pre> 'b',</pre></td>
      <td><pre> 'c',</pre></td>
      <td><pre> 'd',</pre></td>
      <td class="underline-cell"><pre> 'e',</pre></td>
      <td><pre> 'f',</pre></td>
      <td><pre> 'g']</pre></td>
    </tr>
    <tr>
      <th>index</th>
      <td></td>
      <td class="slice-diagram-not-selected">&minus;7</td>
      <td class="slice-diagram-not-selected">&minus;6</td>
      <td class="slice-diagram-not-selected">&minus;5</td>
      <td class="slice-diagram-not-selected">&minus;4</td>
      <td class="slice-diagram-selected">&minus;3</td>
      <td class="slice-diagram-not-selected">&minus;2</td>
      <td class="slice-diagram-not-selected">&minus;1</td>
    </tr>
  </table>
</div>

```py
>>> a[-3]
'e'
```

An equivalent way to think about negative indices is that an index
`a[-i]` selects `a[len(a) - i]`, that is, you can subtract the negative
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

For NumPy arrays, `i` is bounded by the size of the axis being indexed (not
the total size of the array):


```py
>>> import numpy as np
>>> a = np.ones((2, 3)) # A has 6 elements but the first axis has size 2
>>> a[2]
Traceback (most recent call last):
...
IndexError: index 2 is out of bounds for axis 0 with size 2
>>> a[-3]
Traceback (most recent call last):
...
IndexError: index -3 is out of bounds for axis 0 with size 2
```

Fortunately, NumPy arrays give more helpful `IndexError` error messages than
Python lists do.

The second important fact about integer indexing is that it reduces the
dimensionality of the container being indexed. For a `list` or `tuple`, this
means that an integer index returns an element of the list, which is in
general a different type than `list` or `tuple`. For instance, above we saw
that indexing `a` with an integer resulted in a `str`, because `a` is a list
that contains strings. This is in contrast with [slices](slices.md), which
always [return the same container type](subarray).

(strings-integer-indexing)=
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
[multidimensional index](multidimensional-indices/index). The result is always
an array with the dimensionality reduced by 1, namely, the axis being indexed
is removed from the resulting shape. This is in contrast with
[slices](slices.md), which always [maintain the dimension being
sliced](subarray).

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

If `a` has only a single dimension, the result is a 0-D array, i.e., a single
scalar element (just as if `a` were a list):

```py
>>> a = np.asarray(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
>>> a[3] # doctest: +SKIPNP1
np.str_('d')
```

In general, the resulting array is a subarray corresponding to the `i`-th
position along the given axis, using the 0- and &minus;1-based rules discussed
above. For example:

```py
>>> a = np.arange(4).reshape((2, 2))
>>> a
array([[0, 1],
       [2, 3]])
>>> a[0] # The first subarray along the first axis
array([0, 1])
>>> a[1] # The second subarray along the first axis
array([2, 3])
>>> a[:, 0] # The first subarray along the second axis
array([0, 2])
>>> a[:, 1] # The second subarray along the second axis
array([1, 3])
```

A helpful analogy for understanding integer indexing on NumPy arrays is to
consider it in terms of a [list of
lists](multidimensional-indices/what-is-an-array.md). An integer index on the
first axis `a[i]` selects the `i`-th sub-list at the top level of sub-list
nesting. And in general, an integer index `i` on axis `k` selects the `i`-th
sub-lists at the `k`-th nesting level.[^nesting-level] For example, if `l` is
a nested list of lists

[^nesting-level]: Thinking about the `k`-th level of nesting can get
    confusing. For instance, it is unclear whether `k` should be counted with
    0-based or 1-based numbering, or which level counts as which, considering
    that at the outermost "level," there is always a single list.
    List-of-lists is a good analogy for thinking about why one might want to
    use an nd-array in the first place, but as you actually use NumPy arrays
    in practice, you'll find it's much better to think about dimensions and
    axes directly, not "levels of nesting."

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

```{rubric} Footnotes
```
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
