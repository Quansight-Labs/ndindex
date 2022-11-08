===============
 API Reference
===============

The ndindex API consists of classes representing the different types of index
objects (integers, slices, etc.), as well as some helper functions for dealing
with indices.


ndindex
=======

.. autofunction:: ndindex.ndindex

.. _index-types:

Index Types
===========

The following classes represent different types of indices.

.. _integer-api:

.. autoclass:: ndindex.Integer
   :members:
   :special-members:

.. _slice-api:

.. autoclass:: ndindex.Slice
   :members:
   :special-members:

.. _ellipsis-api:

.. autoclass:: ndindex.ellipsis
   :members:

.. _newaxis-api:

.. autoclass:: ndindex.Newaxis
   :members:

.. _tuple-api:

.. autoclass:: ndindex.Tuple
   :members:

.. _integerarray-api:

.. autoclass:: ndindex.IntegerArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:

.. _booleanarray-api:

.. autoclass:: ndindex.BooleanArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:

Index Helpers
=============

The functions here are helpers for working with indices that aren't methods of
the index objects.

.. autofunction:: ndindex.iter_indices

.. autoexception:: ndindex.BroadcastError

.. autoexception:: ndindex.AxisError

Chunking
========

ndindex contains objects to represent chunking an array.

.. autoclass:: ndindex.ChunkSize
   :members:

Internal API
============

These classes are only intended for internal use in ndindex. They shouldn't
relied on as they may be removed or changed.

.. autoclass:: ndindex.ndindex.ImmutableObject
   :members:

.. autoclass:: ndindex.ndindex.NDIndex
   :members:

.. autoclass:: ndindex.array.ArrayIndex
   :members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation: Subclasses should redefine this

.. autoclass:: ndindex.slice.default

.. autofunction:: ndindex.ndindex.asshape

.. autofunction:: ndindex.ndindex.operator_index

.. autofunction:: ndindex.ndindex.ncycles

.. autofunction:: ndindex.ndindex.broadcast_shapes
