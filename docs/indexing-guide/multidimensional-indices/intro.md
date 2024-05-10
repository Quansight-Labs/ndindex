(what-is-an-array)=
# Introduction: What is an Array?

Before we look at indices, let's take a step back and look at the NumPy array.
Just what is it that makes NumPy arrays so ubiquitous and NumPy one of the
most successful numerical tools ever? The answer is quite a few things, which
come together to make NumPy a fast and easy to use library for array
computations. But one feature in particular stands out: multidimensional
indexing.

Let's consider pure Python for a second. Suppose we have a list of values, for
example, a list of your bowling scores.

```py
>>> scores = [70, 65, 71, 80, 73]
```

From what we learned before, we can now index this list with
[integers](../integer-indices.md) or [slices](../slices.md) to get some
subsets of it.

```py
>>> scores[0] # The first score
70
>>> scores[-3:] # The last three scores
[71, 80, 73]
```

You can imagine all sorts of different things you'd want to do with your
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
change your `scores` list into a list of lists. The first inner list is your
scores, and the second will be Bob's:

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
the expression `[i[0] for i in l]` is invalid, because not every element
of `l` is a list.

NumPy arrays function like a list of lists, but are restricted so that these
kinds of operations always "make sense". More specifically, if you have a
"list of lists", each element of the "outer list" must be a list. `[1, [2,
3]]` is not a valid NumPy array. Furthermore, each inner list must have the
same length, or more precisely, the lists at each level of nesting must have
the same length.

Lists of lists can be nested more than just two levels deep. For example, you
might want to take your scores and create a new outer list, splitting them by
season. Then you would have a list of lists of lists, and your indexing
operations would look like `[[game[0] for game in season] for season in
scores]`.

In NumPy, these different levels of nesting are called *axes* or *dimensions*.
The number of axes---the level of nesting---is called the
*rank*[^rank-footnote] or *dimensionality* of the array. Together, the lengths
of these lists at each level are called the *shape* of the array (remember
that the lists at each level have to have the same number of elements).

[^rank-footnote]: Not to be confused with [mathematical definitions of
    rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)). Because of
    this ambiguity, the term "dimensionality" or "number of dimensions" is
    generally preferred to "rank", and is what we use in this guide.

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

This index contains two components: the slice `:` and the integer index `0`.
The slice `:` says to take everything from the first axis (which represents
games), and the integer index `0` says to take the first element of the second
axis (which represents people).

The shape of our array is a tuple with the number of games (the outer axis)
and the number of people (the inner axis).

```py
>>> scores.shape
(5, 2)
```

This is the power of multidimensional indexing in NumPy arrays. If we have a
list of lists of numbers, or a list of lists of lists of numbers, or a list of
lists or lists of lists..., we can index things at any "nesting level" equally
easily. There is a small reasonable restriction, namely that each "level" of
lists (axis) must have the same number of elements. This restriction is
reasonable because in the real world, data tends to be tabular, like bowling
scores, meaning each axis will naturally have the same number of elements.
Even if this weren't the case, for instance, if Bob were out sick for a game,
we could easily use a sentinel value like `-1` or `nan` for a missing value to
maintain uniform lengths.

The indexing semantics are only a small part of what makes NumPy arrays so
powerful. They also have many other advantages that are unrelated to indexing.
They operate on contiguous memory using native machine data types, which makes
them very fast. They can be manipulated using array expressions with
[broadcasting semantics](broadcasting); for example, you can easily add a
handicap to the scores array with something like `scores + np.array([124,
95])`, which would itself be a nightmare using the list of lists
representation. This, along with the powerful ecosystem of libraries like
`scipy`, `matplotlib`, and thousands of others, are what have made NumPy such
a popular and essential tool.

## Footnotes
<!-- Footnotes are written inline above but markdown will put them here at the
end of the document. -->
