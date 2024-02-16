===============
 API Reference
===============

The ndindex API consists of classes representing the different types of index
objects (integers, slices, etc.), as well as some helper functions for dealing
with indices.


ndindex
=======

The primary entry-point to the ndindex API is the `ndindex()` function, which
converts Python index objects into ndindex objects.

.. autofunction:: ndindex.ndindex

API Reference Index
===================

.. toctree::
   :titlesonly:

   index-types.rst
   shapetools.rst
   chunking.rst
   exceptions.rst
   internal.rst
