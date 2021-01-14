from .ndindex import NDIndex, asshape, operator_index

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

    Note that `Integer` itself implements `__index__`, so it can be used as an
    index directly. However, it is still recommended to use `raw` for
    consistency, as this only works for `Integer`.

    .. note::

       `Integer` does *not* represent an integer, but rather an
       *integer index*. It does not have most methods that `int` has, and
       should not be used in non-indexing contexts. See the document on
       :any:`type-confusion` for more details.

    """
    def _typecheck(self, idx):
        idx = operator_index(idx)
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
        Reduce an Integer index on an array of shape `shape`.

        The result will either be `IndexError` if the index is invalid for the
        given shape, or an Integer index where the value is nonnegative.

        >>> from ndindex import Integer
        >>> idx = Integer(-5)
        >>> idx.reduce((3,))
        Traceback (most recent call last):
        ...
        IndexError: index -5 is out of bounds for axis 0 with size 3
        >>> idx.reduce((9,))
        Integer(4)

        See Also
        ========

        .NDIndex.reduce
        .Tuple.reduce
        .Slice.reduce
        .ellipsis.reduce
        .Newaxis.reduce
        .IntegerArray.reduce
        .BooleanArray.reduce

        """
        if shape is None:
            return self

        shape = asshape(shape, axis=axis)
        size = shape[axis]
        if self.raw >= size or -size > self.raw < 0:
            raise IndexError(f"index {self.raw} is out of bounds for axis {axis} with size {size}")

        if self.raw < 0:
            return self.__class__(size + self.raw)

        return self

    def newshape(self, shape):
        # The docstring for this method is on the NDIndex base class
        shape = asshape(shape)

        # reduce will raise IndexError if it should be raised
        self.reduce(shape)
        return shape[1:]

    def as_subindex(self, index):
        from .ndindex import ndindex
        from .slice import Slice
        from .tuple import Tuple

        index = ndindex(index)

        if isinstance(index, Tuple):
            return Tuple(self).as_subindex(index)

        if not isinstance(index, Slice):
            raise NotImplementedError("Integer.as_subindex is only implemented for slices")

        if self == -1:
            s = Slice(self.args[0], None).as_subindex(index)
        else:
            s = Slice(self.args[0], self.args[0] + 1).as_subindex(index)
        if s == Slice(0, 0, 1):
            # The intersection is empty. There is no valid index we can return
            # here. We want an index that produces an empty array, but the
            # shape should be one less, to match a[self]. Since a[index] has
            # as many dimensions as a, there is no way to index a[index] so
            # that it gives one fewer dimension but is also empty. The best we
            # could do is to return a boolean array index array([False]),
            # which would replace the first dimension with a length 0
            # dimension. But
            #
            # 1. this isn't implemented yet,
            # 2. there are complications if this happens in multiple
            #    dimensions (it might not be possible to represent, I'm not
            #    sure), and
            # 3. Slice.as_subindex(Integer) also raises this exception in the
            #    case of an empty intersection (see the comment in that code).
            raise ValueError(f"{self} and {index} do not intersect")
        assert len(s) == 1
        return Integer(s.args[0])

    def isempty(self, shape=None):
        if shape is not None:
            return 0 in self.newshape(shape)

        return False

    def __eq__(self, other):
        if isinstance(other, Integer):
            return self.args == other.args
        try:
            other = operator_index(other)
        except TypeError:
            return False
        return self.args[0] == other

    def __hash__(self):
        return super().__hash__()
