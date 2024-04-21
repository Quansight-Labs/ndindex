# Guide to NumPy Indexing

This section of the ndindex documentation discusses the semantics of NumPy
indices. This really is more of a documentation of NumPy itself than of
ndindex. However, understanding the underlying semantics of indices is
critical to making the best use of ndindex, as well as for making the best use
of NumPy arrays themselves. Furthermore, the sections on [integer
indices](integer-indices) and [slices](slices-docs) also apply to the built-in
Python sequence types like `list` and `str`.

This guide is aimed for people who are new to NumPy indexing semantics, but it
also tries to be as complete as possible and at least mention all the various
corner cases. Some of these technical points can be glossed over if you are a
beginner.

## Table of Contents

This guide is split into four sections.

After a short [introduction](intro.md), the first two sections cover the basic
single-axis index types: [integer indices](integer-indices.md), and
[slices](slices.md). These are the indices that only work on a single axis of
an array at a time. These are also the indices that work on built-in sequence
types such as `list` and `str`. The semantics of these index types on `list`
and `str` are exactly the same as on NumPy arrays, so even if you do not care
about NumPy or array programming, these sections of this document can be
informative just as a general Python programmer. Slices in particular are oft
confused and the guide on slicing clarifies their exact rules and debunks some
commonly spouted false beliefs about how they work.

The third section covers [multidimensional
indices](multidimensional-indices.md). These indices will not work on the
built-in Python sequence types like `list` and `str`; they are only defined
for NumPy arrays.

Finally, a page on [other topics relevant to indexing](other-topics.md) covers
a set of miscellaneous topics about NumPy arrays that are useful for
understanding how indexing works, such as [broadcasting](broadcasting),
[views](views-vs-copies), [strides](strides), and
[ordering](c-vs-fortran-ordering).

```{toctree}
:titlesonly:

intro.md
integer-indices.md
slices.md
multidimensional-indices.md
other-topics.md
```
