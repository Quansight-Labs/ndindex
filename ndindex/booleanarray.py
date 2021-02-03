from numpy import bool_, count_nonzero, ndarray

from .array import ArrayIndex
from .ndindex import asshape

class BooleanArray(ArrayIndex):
    """
    Represents a boolean array index (also known as a mask).

    If `idx` is an n-dimensional boolean array with shape `s = (s1, ..., sn)`
    and `a` is an array of shape `s = (s1, ..., sn, ..., sm)`, `a[idx]`
    replaces the first `n` dimensions of `a` with a single dimensions of size
    `np.nonzero(idx)`, where each entry is included if the corresponding
    element of `idx` is True. The axes in the index shape should match the
    corresponding axes in the array shape or be 0, or the index produces
    IndexError.

    The typical way of creating a mask is to use boolean operations on an
    array, then index the array with that. For example, if `a` is an array of
    integers, `a[a > 0]` will produces a flat array of the elements of `a`
    that are positive.

    Some important things to note about boolean array index semantics:

    1. A boolean array index will remove as many dimensions as the index has,
       and replace them with a single flat dimension which is the size of the
       number of `True` elements in the index.

    2. A boolean array index `idx` works the same as the integer array index
       `np.nonzero(idx)`. In particular, the elements of the index are always
       iterated in row-major, C-style order. This does not apply to
       0-dimensional boolean indices.

    3. A 0-dimensional boolean index (i.e., just the scalar `True` or `False`)
       can still be thought of as removing 0 dimensions and adding a single
       dimension of length 1 for True or 0 for False. Hence, if `a` has shape
       `(s1, ..., sn)`, then `a[True]` has shape `(1, s1, ..., sn)`, and
       `a[False]` has shape `(0, s1, ..., sn)`.

    4. If a tuple index has multiple boolean arrays, they are broadcast
       together and iterated as a single array, similar to
       :class:`IntegerArray`. If a boolean array index `idx` is mixed with an
       integer array index in a tuple index, it is treated like
       `np.nonzero(idx)`.

    A list of booleans may also be used in place of a boolean array. Note
    that NumPy treats a direct list of integers as a tuple index, but this
    behavior is deprecated and will be replaced with integer array indexing in
    the future. ndindex always treats lists as arrays.

    >>> from ndindex import BooleanArray
    >>> import numpy as np
    >>> idx = BooleanArray([[ True,  True],
    ...                     [ True, False],
    ...                     [False, False],
    ...                     [False,  True],
    ...                     [False, False]])
    >>> a = np.arange(10).reshape((5, 2))
    >>> a[idx.raw]
    array([0, 1, 2, 7])

    .. note::

       `BooleanArray` does *not* represent an array, but rather an *array
       index*. It does not have most methods that `numpy.ndarray` has, and
       should not be used in array contexts. See the document on
       :any:`type-confusion` for more details.

    """

    dtype = bool_
    """
    The dtype of `BooleanArray` is `np.bool_`.
    """

    def __hash__(self):
        # Match the hash for scalar booleans. Otherwise, hash(True) won't
        # equal hash(ndindex(True)).
        if self.shape == ():
            return hash(self.array.any())
        return super().__hash__()

    @property
    def count_nonzero(self):
        """
        Returns the number of elements indexed by self.

        In general, if shapes match, when indexed by `self`, the first *n*
        dimensions of an array are replaced with a single dimension of size
        `count_nonzero`, where *n* is `self.shape`.

        This is the same as `np.count_nonzero(self.array)`. Note, to get the
        shape of an array indexed by self, use :meth:`newshape`, not this
        method.

        >>> from ndindex import BooleanArray
        >>> BooleanArray([True, False, True]).count_nonzero
        2
        """
        return count_nonzero(self.array)

    def reduce(self, shape=None, axis=0):
        """
        Reduce a `BooleanArray` index on an array of shape `shape`.

        The result will either be `IndexError` if the index is invalid for the
        given shape, or a `BooleanArray` index. Presently, no simplifications
        are done for BooleanArray: if `reduce()` does not produce an
        `IndexArray` the index returned will be the same as `self`.

        >>> from ndindex import BooleanArray
        >>> idx = BooleanArray([True, False])
        >>> idx.reduce((3,))
        Traceback (most recent call last):
        ...
        IndexError: boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 2
        >>> idx.reduce((2,))
        BooleanArray([True, False])

        See Also
        ========

        .NDIndex.reduce
        .Tuple.reduce
        .Slice.reduce
        .ellipsis.reduce
        .Newaxis.reduce
        .Integer.reduce
        .IntegerArray.reduce

        """
        if shape is None:
            return self

        shape = asshape(shape)

        if len(shape) < self.ndim + axis:
            raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional, but {self.ndim + axis} were indexed")

        for i in range(axis, axis+self.ndim):
            if self.shape[i-axis] != 0 and shape[i] != self.shape[i-axis]:

                raise IndexError(f"boolean index did not match indexed array along dimension {i}; dimension is {shape[i]} but corresponding boolean dimension is {self.shape[i-axis]}")

        return self

    def newshape(self, shape):
        # The docstring for this method is on the NDIndex base class
        shape = asshape(shape)

        # reduce will raise IndexError if it should be raised
        self.reduce(shape)
        return (self.count_nonzero,) + shape[self.ndim:]

    def isempty(self, shape=None):
        if shape is not None:
            return 0 in self.newshape(shape)

        return self.count_nonzero == 0

    def as_subindex(self, index):
        from .tuple import Tuple

        if self in [True, False]:
            raise NotImplementedError("as_subindex is not supported for scalar boolean indices")
        return Tuple(*self.array.nonzero()).as_subindex(index)

    def broadcast_arrays(self):
        from .tuple import Tuple
        return Tuple(self).broadcast_arrays()

    def __eq__(self, other):
        if isinstance(other, (bool, bool_)):
            return self.shape == () and self.array == other
        if isinstance(other, BooleanArray):
            b = other.array
        elif isinstance(other, ndarray):
            b = other
        elif isinstance(other, list):
            try:
                b = BooleanArray(other)
            except TypeError:
                return False
        else:
            return False
        a = self.array
        return a.shape == b.shape and (a == b).all()

def _is_boolean_scalar(idx):
    """
    Determine if idx is a scalar boolean index.

    This is for internal usage only. Assumes idx is already an ndindex type.
    This is more performant than `idx in [True, False]`.
    """
    # TODO: Instead of this function, make BooleanScalar a separate class.
    return isinstance(idx, BooleanArray) and idx.shape == ()
