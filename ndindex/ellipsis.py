from .ndindex import NDIndex
from .tuple import Tuple

class ellipsis(NDIndex):
    """
    Represents an ellipsis index, i.e., `...` (or `Ellipsis`).

    Ellipsis indices by themselves return the full array. Inside of a tuple
    index, an ellipsis skips 0 or more axes of the array so that everything
    after the ellipsis indexes the last axes of the array. A tuple index can
    have at most one ellipsis.

    For example `a[(0, ..., -2)]` would index the first element on the first
    axis, the second-to-last element in the last axis, and include all the
    axes in between.

    >>> from numpy import arange
    >>> a = arange(2*3*4).reshape((2, 3, 4))
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> a[0, ..., -2]
    array([ 2,  6, 10])

    An ellipsis can go at the beginning of end of a tuple index, and is
    allowed to match 0 axes.

    **Note:** Unlike the standard Python `Ellipsis`, `ellipsis` is the type,
    not the object (the name is lowercase to avoid conflicting with the
    built-in). Use `ellipsis()` or `ndindex(...)` to create the object. Also
    unlike `Ellipsis`, `ellipsis()` is not singletonized, so you should not
    use `is` to compare it.

    """
    def _typecheck(self):
        return ()

    def reduce(self, shape=None):
        """
        Reduce an ellipsis index

        Since an ellipsis by itself always returns the full array unchanged,
        `ellipsis().reduce()` returns `Tuple()` as a canonical form (the index
        `()` also always returns an array unchanged).

        >>> from ndindex import ellipsis
        >>> ellipsis().reduce()
        Tuple()

        """
        return Tuple()

    @property
    def raw(self):
        return ...
