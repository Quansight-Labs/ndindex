# API Reference

The ndindex API consists of classes representing the different types of index
objects (integers, slices, etc.), as well as some helper functions for dealing
with indices.

## ndindex

### ndindex

```eval_rst
.. autofunction:: ndindex.ndindex

```

### Integer

```eval_rst
.. autoclass:: ndindex.Integer
   :members:
```

### Slice

```eval_rst
.. autoclass:: ndindex.Slice
   :members:
```

### ellipsis

```eval_rst
.. autoclass:: ndindex.ellipsis
   :members:
```

### Tuple

```eval_rst
.. autoclass:: ndindex.Tuple
   :members:
```


### Internal API

These classes are only intended for internal use in ndindex.

```eval_rst
.. autoclass:: ndindex.ndindex.NDIndex
   :members:

.. autoclass:: ndindex.slice.default

```
