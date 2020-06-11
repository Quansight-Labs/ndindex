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

Integer
-------

.. autoclass:: ndindex.Integer
   :members:

.. _slice-api:

Slice
-----

.. autoclass:: ndindex.Slice
   :members:

ellipsis
--------

.. autoclass:: ndindex.ellipsis
   :members:

Tuple
-----

.. autoclass:: ndindex.Tuple
   :members:

Internal API
------------

These classes are only intended for internal use in ndindex.

.. autoclass:: ndindex.ndindex.NDIndex
   :members:

.. autoclass:: ndindex.slice.default
