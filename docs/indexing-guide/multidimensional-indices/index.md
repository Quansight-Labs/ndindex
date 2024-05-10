# Multidimensional Indices

Unlike [integers](../integer-indices.md) and [slices](../slices.md), which not
only work on NumPy arrays but also on built-in Python sequence types such as
`list`, `tuple`, and `str`, the remaining index types do not work at all on
built-in sequence types. For example, if you try to use one of the index types
described on this page on a `list`, you will get an `IndexError` The semantics
of these indices are defined by the NumPy library, not the Python language.

To begin, we should be sure we understand what an array is:

- [](what-is-an-array.md)

(basic-indices)=
## Basic Multidimensional Indices

First, let's look at the basic multidimensional indices ("basic" as opposed to
["advanced" indices](advanced-indices), which are discussed below). We've
already learned about two in previous sections:

- [](../integer-indices.md)
- [](../slices.md)

There are three others:

- [](tuples.md)
- [](ellipses.md)
- [](newaxis.md)

(advanced-indices)=
## Advanced Indices

Finally, we come to the so-called advanced indices. These are "advanced" in
the sense that they are more complex. They are also distinct from "basic"
indices in that they always return a copy (see [](views-vs-copies)). Advanced
indices allow selecting arbitrary parts of an array, in ways that are
impossible with the basic index types. Advanced indexing is also sometimes
called "fancy indexing" or indexing by arrays, as the indices themselves are
arrays:

- [](integer-arrays.md)
- [](boolean-arrays.md)

Using an array that does not have an integer or boolean dtype as an index
results in an error.

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
