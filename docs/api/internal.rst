============
Internal API
============

These classes are only intended for internal use in ndindex. They shouldn't
relied on as they may be removed or changed.

Note that the documentation for methods on ndindex classes will sometimes link
to this page because the methods are defined on the on the
:class:`~.ImmutableObject`, :class:`~.NDIndex`, or :class:`~.ArrayIndex` base
classes. These classes are not designed to be used directly. Such methods are
present on all `ndindex classes <index-types.rst>`_, which are what should be
actually be constructed. Remember that the primary entry-point API for
constructing ndindex index classes is the :func:`~.ndindex` function.

Base Classes
============

.. autoclass:: ndindex.ndindex.ImmutableObject
   :members:

.. autoclass:: ndindex.ndindex.NDIndex
   :members:

.. autoclass:: ndindex.array.ArrayIndex
   :members:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation: Subclasses should redefine this

Other Internal Functions
========================

.. autoclass:: ndindex.slice.default

.. autofunction:: ndindex.ndindex.operator_index

.. autofunction:: ndindex.shapetools.asshape

.. autofunction:: ndindex.shapetools.ncycles

.. autofunction:: ndindex.shapetools.associated_axis

.. autofunction:: ndindex.shapetools.remove_indices

.. autofunction:: ndindex.shapetools.unremove_indices

.. autofunction:: ndindex.shapetools.normalize_skip_axes
