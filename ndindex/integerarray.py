import warnings

from numpy import (ndarray, asarray, integer, bool_, intp, array2string,
                   empty, zeros)

from .ndindex import NDIndex, asshape

class IntegerArray(NDIndex):
    """
    Represents an integer array.

    If `idx` is an n-dimensional integer array with shape `s = (s1, ..., sn)`
    and `a` is any array, `a[idx]` replaces the first dimension of `a` with
    `s1, ..., sn` dimensions, where each entry is indexed according to the
    entry in `idx` as an integer index.

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

    """
    def _typecheck(self, idx, shape=None):
        if shape is not None:
            if idx != []:
                raise ValueError("The shape argument is only allowed for empty arrays (idx=[])")
            shape = asshape(shape)
            if 0 not in shape:
                raise ValueError("The shape argument must be an empty shape")
            idx = empty(shape, dtype=intp)

        if isinstance(idx, (list, ndarray, bool, integer, int, bool_)):
            # Ignore deprecation warnings for things like [1, []]. These will be
            # filtered out anyway since they produce object arrays.
            with warnings.catch_warnings(record=True):
                a = asarray(idx)
                if a is idx:
                    a = a.copy()
                if isinstance(idx, list) and 0 in a.shape:
                    a = a.astype(intp)
            if issubclass(a.dtype.type, integer):
                if a.dtype != intp:
                    a = a.astype(intp)
                a.flags.writeable = False
                return (a,)
            if a.dtype == bool_:
                raise TypeError("Boolean array passed to IntegerArray. Use BooleanArray instead.")
            raise TypeError("The input array must have an integer dtype.")
        raise TypeError("IntegerArray must be created with an array of integers")

    @property
    def raw(self):
        return self.args[0]

    @property
    def array(self):
        """
        Return the NumPy array of self.

        This is the same as `self.args[0]`.
        """
        return self.args[0]

    @property
    def shape(self):
        """
        Return the shape of the array of self.

        This is the same as self.array.shape. Note that this is **not** the
        same as the shape of an array that is indexed by self. Use
        :meth:`newshape` to get that.

        """
        return self.array.shape

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
        .Integer.reduce

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

    # The repr form recreates the object. The str form gives the truncated
    # array string and is explicitly non-valid Python (doesn't have commas).
    def __repr__(self):
        if 0 not in self.shape:
            arg = repr(self.array.tolist())
        else:
            arg = f"[], shape={self.shape}"
        return f"{self.__class__.__name__}({arg})"

    def __str__(self):
        return (self.__class__.__name__
                + "("
                + array2string(self.array).replace('\n', '')
                + ")")

    def __hash__(self):
        return hash(self.array.tobytes())
