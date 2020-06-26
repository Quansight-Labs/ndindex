# Documentation Style Guide

```{note}
This document is still a work in progress
```

This is a style guide for the ndindex documentation.

## English Conventions

- Use American English spelling for all documentation.
- The Oxford comma should be used.
- "ndindex" is always written in all lowercase, unless referring to the
  `NDIndex` base class.
- The plural of "index" is "indices". "Indexes" should only be used as a verb.
  For example, "in `a[i, j]`, the indices are `i` and `j`. They represent a
  single tuple index `(i, j)` which indexes the array `a`."
- The arguments of a slice should be referred to as "start", "stop", and
  "step", respectively. This matches the argument names and attributes of the
  `Slice` object.
- A generic index variable should be called `idx`.
- A generic slice variable should be called `s`.
- Example array variables should be called `a`.
- The more concise Python notation for indices should be used where it is
  allowed in doctests and code examples, unless not using an ndindex type
  would lead to confusion or ambiguity. For example, `Tuple` always converts
  its arguments to ndindex types, so `Tuple(slice(1, 2), ..., 3)` is preferred
  over `Tuple(Slice(1, 2), ellipsis(), Integer(3))`.
- `...` should be used in place of `Ellipsis` or `ellipsis()` wherever
  possible (this and the previous rule also apply to the code, not just
  documentation examples).
- The above rules are only guides. If following them to the word would lead to
  ambiguity, they should be broken.

## Markup Conventions

- Sphinx docs can be written either in RST or in Markdown. Markdown documents
  use [MyST](https://myst-parser.readthedocs.io/en/latest/). Markdown is
  generally preferred over RST, although in some cases RST is required if
  something isn't supported by MyST.
- Both RST and MyST support cross references. In RST, a cross reference looks
  like ``` :ref:`reference` ```. In MyST, use either `[link text](reference)`
  or ``` {ref}`link text <reference>` ```. See the [MyST
  docs](https://myst-parser.readthedocs.io/en/latest/using/syntax.html) and
  the [RST
  docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html)
  for more details.
- Text in documentation should use `code` (surrounded by single backticks,
  like ``` `code` ```) whenever it refers to a variable or expression in code.
  Note that only single backticks are required even for RST as the default
  role is set to `'code'`.
- Inline mathematical formulas can be formatted with single dollar signs, like
  `$x^2 + 1$`, which creates $x^2 + 1$. Display formulas, which appear on
  their own line, should use the `.. math::` directive for RST or ````
  ```{math} ```` for Markdown.
- Docstrings are currently written in RST. We may move to Markdown at some
  point. Docstrings use the napoleon extension, meaning they can be written in
  numpydoc format.
- All public docstrings should include a doctest. Doctests can be run with the
  `./run_doctests` script at the root of the repo. This also runs doctests in
  the RST and Markdown files. Doctests are configured so that each function or
  method must import all names used in that function or method.
