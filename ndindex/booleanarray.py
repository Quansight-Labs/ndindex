from numpy import bool_, zeros

from .array import ArrayIndex
from .ndindex import asshape

class BooleanArray(ArrayIndex):
    """
    Represents a boolean array index (also known as a mask).

    If `idx` is an n-dimensional boolean array with shape `s = (s1, ..., sn)`
    and `a` is an array of shape `s = (s1, ..., sn, ..., sm), `a[idx]`
    replaces the first `n` dimensions of `a` with a single dimensions of size
    `np.nonzero(idx)`, where each entry is included if the corresponding
    element of `idx` is True.

    The typical way of creating a mask is to use boolean operations on an
    array, then index the array with that. For example, if `a` is an array of
    integers, `a[a > 0]` will produces a flat array of the elements of `a`
    that are positive.

    Some important things to note about boolean array index semantics:

    1. A boolean array index will remove as many dimensions as the index has,
       and replace them with a single flat dimension which is the size of the
       number of `True` elements in the index.

    2. A boolean array index `idx` works the same as the integer index
       `np.nonzero(idx)`. In particular, the elements of the index are always
       iterated in row-major, C-style order. This does not apply to
       0-dimensional boolean indices.

    3. A 0-dimension boolean index (i.e., just the scalar `True` or `False`)
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
    array([[0, 1],
           [1, 2]])

    .. note::

       `BooleanArray` does *not* represent an array, but rather an *array
       index*. It does not have most methods that `numpy.ndarray` has, and
       should not be used in array contexts. See the document on
       :ref:`type-confusion` for more details.

    """
    dtype = bool_
