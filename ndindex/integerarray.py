from numpy import intp, zeros

from .array import ArrayIndex
from .ndindex import asshape

class IntegerArray(ArrayIndex):
    """
    Represents an integer array index.

    If `idx` is an n-dimensional integer array with shape `s = (s1, ..., sn)`
    and `a` is any array, `a[idx]` replaces the first dimension of `a` with
    dimensions of size `s1, ..., sn`, where each entry is indexed according to
    the entry in `idx` as an integer index.

    Integer arrays can also appear as part of tuple indices. In that case,
    they replace the axis being indexed. If more than one integer array
    appears inside of a tuple index, they are broadcast together.

    A list of integers may also be used in place of an integer array. Note
    that NumPy treats a direct list of integers as a tuple index, but this
    behavior is deprecated and will be replaced with integer array indexing in
    the future. ndindex always treats lists as arrays.

    >>> from ndindex import IntegerArray
    >>> import numpy as np
    >>> idx = IntegerArray([[0, 1], [1, 2]])
    >>> a = np.arange(10)
    >>> a[idx.raw]
    array([[0, 1],
           [1, 2]])

    .. note::

       `IntegerArray` does *not* represent an array, but rather an *array
       index*. It does not have most methods that `numpy.ndarray` has, and
       should not be used in array contexts. See the document on
       :ref:`type-confusion` for more details.

    """
    dtype = intp
    """
    The dtype of `IntegerArray` is `np.intp`, which is typically either
    `np.int32` or `np.int64` depending on the platform.
    """

    def reduce(self, shape=None, axis=0):
        """
        Reduce an `IntegerArray` index on an array of shape `shape`.

        The result will either be `IndexError` if the index is invalid for the
        given shape, or an `IntegerArray` index where the values are all
        nonnegative.

        >>> from ndindex import IntegerArray
        >>> idx = IntegerArray([-5, 2])
        >>> idx.reduce((3,))
        Traceback (most recent call last):
        ...
        IndexError: index -5 is out of bounds for axis 0 with size 3
        >>> idx.reduce((9,))
        IntegerArray([4, 2])

        See Also
        ========

        .NDIndex.reduce
        .Tuple.reduce
        .Slice.reduce
        .ellipsis.reduce
        .Newaxis.reduce
        .Integer.reduce
        .BooleanArray.reduce

        """
        from .integer import Integer

        if self.shape == ():
            return Integer(self.array).reduce(shape, axis=axis)

        if shape is None:
            return self

        shape = asshape(shape, axis=axis)
        if 0 in shape[:axis] + shape[axis+1:]:
            # There are no bounds checks for empty arrays if one of the
            # non-indexed axes is 0. This behavior will be deprecated in NumPy
            # 1.20. Once 1.20 is released, we will change the ndindex behavior
            # to match it, since we want to match all post-deprecation NumPy
            # behavior. But it is impossible to test against the
            # post-deprecation behavior reliably until a version of NumPy is
            # released that raises the deprecation warning, so for now, we
            # just match the NumPy 1.19 behavior.
            return IntegerArray(zeros(self.shape, dtype=intp))

        size = shape[axis]
        new_array = self.array.copy()
        out_of_bounds = (new_array >= size) | ((-size > new_array) & (new_array < 0))
        if out_of_bounds.any():
            raise IndexError(f"index {new_array[out_of_bounds].flat[0]} is out of bounds for axis {axis} with size {size}")

        new_array[new_array < 0] += size
        return IntegerArray(new_array)

    def newshape(self, shape):
        # The docstring for this method is on the NDIndex base class
        shape = asshape(shape)

        # reduce will raise IndexError if it should be raised
        self.reduce(shape)

        return self.shape + shape[1:]

    def isempty(self, shape=None):
        if shape is not None:
            return 0 in self.newshape(shape)

        return 0 in self.shape
