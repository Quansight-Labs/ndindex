import operator

from .ndindex import NDIndex

class Integer(NDIndex):
    """
    Represents an integer index on an axis of an nd-array.

    Any object that implements `__index__` can be used as an integer index.

    >>> from ndindex import Integer
    >>> idx = Integer(1)
    >>> [0, 1, 2][idx.raw]
    1
    >>> idx = Integer(-3)
    >>> [0, 1, 2][idx.raw]
    0

    Note that Integer itself implements `__index__`, so it can be used as an
    index directly. However, it is still recommended to use `raw` for
    consistency, as this only works for Integer.

    """
    def _typecheck(self, idx):
        idx = operator.index(idx)

        return (idx,)

    def __index__(self):
        return self.raw

    @property
    def raw(self):
        return self.args[0]

    def __len__(self):
        """
        Returns the number of elements indexed by `self`

        Since `self` is an integer index, this always returns 1. Note that
        integer indices always remove an axis.
        """
        return 1

    def reduce(self, shape=None, axis=0):
        """
        Reduce an Integer index on an array of shape `shape`

        The result will either be IndexError if the index is invalid for the
        given shape, or an Integer index where the value is nonnegative.

        >>> from ndindex import Integer
        >>> idx = Integer(-5)
        >>> idx.reduce((3,))
        Traceback (most recent call last):
        ...
        IndexError: index -5 is out of bounds for axis 0 with size 3
        >>> idx.reduce((9,))
        Integer(4)

        """
        if shape is None:
            return self

        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) <= axis:
            raise IndexError("too many indices for array")

        size = shape[axis]
        if self.raw >= size or -size > self.raw < 0:
            raise IndexError(f"index {self.raw} is out of bounds for axis {axis} with size {size}")

        if self.raw < 0:
            return self.__class__(size + self.raw)

        return self
