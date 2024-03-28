.. _index-types:

Index Types
===========

The ndindex API consists of classes to represent the different kinds of NumPy
indices, :class:`~.Integer`, :class:`~.Slice`, :class:`~.ellipsis`,
:class:`~.Newaxis`, :class:`~.Tuple`, :class:`~.IntegerArray`, and
:class:`~.BooleanArray`. Typical usage of ndindex consists of constructing one
of these classes, typically with the :func:`~.ndindex` constructor, then using
the methods on the objects. With a few exceptions, all index classes have the
same set of methods, so that they can be used uniformly regardless of the
actual index type. Consequently, many of the method docstrings below are
duplicated across all the classes. For classes where there is are particular
things of note for a given method, the docstring will be different (for
example, :meth:`.Slice.reduce` notes the specific invariants that the
:meth:`~.NDIndex.reduce()` method applies to :class:`~.Slice` objects). Such
methods will be noted by their "See Also" sections.

.. autoclass:: ndindex.Integer
   :members:
   :special-members:

.. autoclass:: ndindex.Slice
   :members:
   :special-members:

.. autoclass:: ndindex.ellipsis
   :members:

.. autoclass:: ndindex.Newaxis
   :members:

.. autoclass:: ndindex.Tuple
   :members:

.. autoclass:: ndindex.IntegerArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:

.. autoclass:: ndindex.BooleanArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:
