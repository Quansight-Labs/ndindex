# Multidimensional Indices

This section of the indexing guide deals with indices that only operate on
NumPy arrays. Unlike [integers](../integer-indices.md) and
[slices](../slices.md), which also work on built-in Python sequence types such
as `list`, `tuple`, and `str`, the remaining index types do not work at all on
built-in sequence types. For example, if you try to use a [tuple
index](tuples.md) on a `list`, you will get an `IndexError` The semantics of
these indices are defined by the NumPy library, not the Python language.

To begin, we should be sure we understand what an array is:

- [](what-is-an-array.md)

(basic-indices)=
## Basic Multidimensional Indices

There are two types of multidimensional indices, basic and advanced indices.
Basic indices are so-called because they are simpler and the most common. They
also are notable because they always return a view (see [](views-vs-copies).

We've already learned about two types of basic indices in previous sections:

- [](../integer-indices.md)
- [](../slices.md)

There are three others:

- [](tuples.md)
- [](ellipses.md)
- [](newaxis.md)

(advanced-indices)=
## Advanced Indices

Lastly are the so-called advanced indices. These are "advanced" in the sense
that they are more complex. They are also distinct from "basic" indices in
that they always return a copy (see [](views-vs-copies)). Advanced indices
allow selecting arbitrary parts of an array, in ways that are impossible with
the basic index types. Advanced indexing is also sometimes called "fancy
indexing" or indexing by arrays, as the indices themselves are arrays.

Using an array that does not have an integer or boolean dtype as an index
results in an error.

- [](integer-arrays.md)
- [](boolean-arrays.md)

```{toctree}
:titlesonly:
:hidden:

what-is-an-array.md
tuples.md
ellipses.md
newaxis.md
integer-arrays.md
boolean-arrays.md

```
