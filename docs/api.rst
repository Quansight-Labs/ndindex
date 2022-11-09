===============
 API Reference
===============

The ndindex API consists of classes representing the different types of index
objects (integers, slices, etc.), as well as some helper functions for dealing
with indices.


ndindex
=======

.. autofunction:: ndindex.ndindex

Index Types
===========

The following classes represent different types of indices.


.. autoclass:: ndindex.Integer
   :members:
   :special-members:

.. _slice-api:

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
