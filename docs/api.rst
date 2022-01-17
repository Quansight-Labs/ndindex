===============
 API Reference
===============

The ndindex API consists of classes representing the different types of index
objects (integers, slices, etc.), as well as some helper functions for dealing
with indices.


ndindex
=======

ndindex
-------

.. autofunction:: ndindex.ndindex

Index Types
===========

The following classes represent different types of indices.

Integer
-------

.. autoclass:: ndindex.Integer
   :members:
   :special-members:

.. _slice-api:

Slice
-----

.. autoclass:: ndindex.Slice
   :members:
   :special-members:

ellipsis
--------

.. autoclass:: ndindex.ellipsis
   :members:


Newaxis
-------

.. autoclass:: ndindex.Newaxis
   :members:

Tuple
-----

.. autoclass:: ndindex.Tuple
   :members:

IntegerArray
------------

.. autoclass:: ndindex.IntegerArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:

BooleanArray
------------

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

iter_indices
------------

.. autofunction:: ndindex.iter_indices

Chunking
========

ndindex contains objects to represent chunking an array.

ChunkSize
---------

.. autoclass:: ndindex.ChunkSize
   :members:

Internal API
============

These classes are only intended for internal use in ndindex. They shouldn't
relied on as they may be removed or changed.

ImmutableObject
---------------

.. autoclass:: ndindex.ndindex.ImmutableObject
   :members:

NDIndex
-------

.. autoclass:: ndindex.ndindex.NDIndex
   :members:

ArrayIndex
----------

.. autoclass:: ndindex.array.ArrayIndex
   :members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation: Subclasses should redefine this

default
-------

.. autoclass:: ndindex.slice.default

asshape
-------

.. autofunction:: ndindex.ndindex.asshape

operator_index
--------------

.. autofunction:: ndindex.ndindex.operator_index

ncycles
-------

.. autofunction:: ndindex.ndindex.ncycles
