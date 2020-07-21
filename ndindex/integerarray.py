import warnings

from numpy import ndarray, asarray, integer, bool_, intp, ndindex as numpy_ndindex

from .ndindex import NDIndex, asshape

class IntegerArray(NDIndex):
    """
    Represents an integer array.

    If `idx` is an n-dimensional integer array with shape `s = (s1, ..., sn)`
    and `a` is any array, `a[idx]` replaces the first axis of `a` with `s`,
    where each entry is indexed according to the entry in `idx`.

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
    def _typecheck(self, idx):
        if isinstance(idx, (list, ndarray, bool)):
            # Ignore deprecation warnings for things like [1, []]. These will be
            # filtered out anyway since they produce object arrays.
            with warnings.catch_warnings(record=True):
                if isinstance(idx, list) and idx == []:
                    a = asarray([], dtype=intp)
                else:
                    a = asarray(idx)
            if issubclass(a.dtype.type, integer):
                if a.dtype != intp:
                    a = a.astype(intp)
                return (a,)
            elif a.dtype == bool_:
                raise TypeError("Boolean array passed to IntegerArray. Use BooleanArray instead.")
            else:
                raise TypeError("The input array must have an integer dtype.")

    @property
    def raw(self):
        return self.args[0]

    @property
    def array(self):
        """
        Return the NumPy array of self.

        This is the same as self.args[0].
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
        from .integer import Integer

        if shape is None:
            return self

        shape = asshape(shape)

        new_array = ndarray(self.shape, dtype=intp)
        for index in numpy_ndindex(self.shape):
            new_array[index] = Integer(self.array[index]).reduce(shape, axis=axis).raw
        return IntegerArray(new_array)
