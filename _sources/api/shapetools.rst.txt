Shape Tools
===========

ndindex contains several helper functions for working with and manipulating
array shapes.

Functions
---------

.. autofunction:: ndindex.iter_indices

.. autofunction:: ndindex.broadcast_shapes

Exceptions
----------

These are some custom exceptions that are raised by the above functions. Note
that most of the other functions in ndindex will raise `IndexError` (if the
index would be invalid), or `TypeError` or `ValueError` (if the input types or
values are incorrect).

.. autoexception:: ndindex.BroadcastError

.. autoexception:: ndindex.AxisError
