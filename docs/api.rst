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
   :special-members:

Tuple
-----

.. autoclass:: ndindex.Tuple
   :members:
   :special-members:

IntegerArray
------------

.. autoclass:: ndindex.IntegerArray
   :members:
   :special-members:


BooleanArray
------------

.. autoclass:: ndindex.BooleanArray
   :members:
   :special-members:

Internal API
------------

These classes are only intended for internal use in ndindex.

.. autoclass:: ndindex.ndindex.NDIndex
   :members:

.. autoclass:: ndindex.slice.default

.. autofunction:: ndindex.ndindex.asshape
