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

.. _index-types:

Index Types
===========

The following classes represent different types of indices.

.. _integer-api:

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

.. _ellipsis-api:

ellipsis
--------

.. autoclass:: ndindex.ellipsis
   :members:

.. _newaxis-api:

Newaxis
-------

.. autoclass:: ndindex.Newaxis
   :members:

.. _tuple-api:

Tuple
-----

.. autoclass:: ndindex.Tuple
   :members:

.. _integerarray-api:

IntegerArray
------------

.. autoclass:: ndindex.IntegerArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:

.. _booleanarray-api:

BooleanArray
------------

.. autoclass:: ndindex.BooleanArray
   :members:
   :inherited-members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation:

Chunking
========

ndindex contains objects to represent chunking an array.

ChunkSize
---------

.. autoclass:: ndindex.ChunkSize
   :members:

Internal API
============

These classes are only intended for internal use in ndindex.

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
