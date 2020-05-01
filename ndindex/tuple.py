from .ndindex import NDIndex, ndindex

class Tuple(NDIndex):
    """
    Represents a tuple of single-axis indices.

    Valid single axis indices are

    - `Integer`
    - `Slice`
    - `ellipsis`
    - `Newaxis`
    - `IntegerArray`
    - `BooleanArray`

    (some of the above are not yet implemented)

    `Tuple(x1, x2, …, xn)` represents the index `a[x1, x2, …, xn]` or,
    equivalently, `a[(x1, x2, …, xn)]`. `Tuple()` with no arguments is the
    empty tuple index, `a[()]`, which returns `a` unchanged.

    >>> from ndindex import Tuple, Slice
    >>> import numpy as np
    >>> idx = Tuple(0, Slice(2, 4))
    >>> a = np.arange(10).reshape((2, 5))
    >>> a
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> a[0, 2:4]
    array([2, 3])
    >>> a[idx.raw]
    array([2, 3])

    """
    def _typecheck(self, *args):
        from .ellipsis import ellipsis

        newargs = []
        for arg in args:
            newarg = ndindex(arg)
            if isinstance(newarg, Tuple):
                raise NotImplementedError("tuples of tuples are not yet supported")
            newargs.append(newarg)

        if newargs.count(ellipsis()) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        return tuple(newargs)

    def __repr__(self):
        # Since tuples are nested, we can print the raw form of the args to
        # make them a little more readable.
        def _repr(s):
            if s is Ellipsis:
                return '...'
            return repr(s)
        return f"{self.__class__.__name__}({', '.join(map(_repr, self.raw))})"

    @property
    def has_ellipsis(self):
        """
        Returns True if self has an ellipsis
        """
        from .ellipsis import ellipsis

        return ellipsis() in self.args

    @property
    def ellipsis_index(self):
        """
        Give the index i of `self.args` where the ellipsis is.

        If `self` doesn't have an ellipsis, it gives `len(self.args)`, since
        tuple indices always implicitly end in an ellipsis.

        The resulting value `i` is such that `self.args[:i]` indexes the
        beginning axes of an array and `self.args[i+1:]` indexes the end axes
        of an array.

        >>> from ndindex import Tuple
        >>> idx = Tuple(0, 1, ..., 2, 3)
        >>> i = idx.ellipsis_index
        >>> i
        2
        >>> idx.args[:i]
        (Integer(0), Integer(1))
        >>> idx.args[i+1:]
        (Integer(2), Integer(3))

        >>> Tuple(0, 1).ellipsis_index
        2

        """
        from .ellipsis import ellipsis

        if self.has_ellipsis:
            return self.args.index(ellipsis())
        return len(self.args)

    @property
    def raw(self):
        return tuple(i.raw for i in self.args)

    def reduce(self, shape=None):
        """
        Reduce an Tuple index on an array of shape `shape`

        A `Tuple` with a single argument is always reduced to that single
        argument (because `a[idx,]` is the same as `a[idx]`).

        >>> from ndindex import Tuple, Integer, Slice

        >>> Tuple(Slice(2, 4)).reduce()
        Slice(2, 4, 1)

        If an explicit array shape is given, The result will either be
        IndexError if the index is invalid for the given shape, or Tuple where
        the entries are recursively reduced.

        >>> idx = Tuple(Slice(0, 10), Integer(-3))
        >>> idx.reduce((5,))
        Traceback (most recent call last):
        ...
        IndexError: too many indices for array
        >>> idx.reduce((5, 2))
        Traceback (most recent call last):
        ...
        IndexError: index -3 is out of bounds for axis 1 with size 2
        >>> idx.reduce((5, 3))
        Tuple(slice(0, 5, 1), 0)

        """
        from .ellipsis import ellipsis

        if len(self.args) == 1:
            return self.args[0].reduce(shape)

        args = self.args
        if args and args[-1] == ellipsis():
            args = args[:-1]

        if shape is None:
            return type(self)(*args)

        if isinstance(shape, int):
            shape = (shape,)
        if (self.has_ellipsis and len(shape) < len(self.args) - 1
            or not self.has_ellipsis and len(shape) < len(self.args)):
            raise IndexError("too many indices for array")

        ellipsis_i = self.ellipsis_index

        newargs = []
        for i, s in enumerate(args[:ellipsis_i]):
            newargs.append(s.reduce(shape, axis=i))

        if ellipsis_i < len(args) and len(args) <= len(shape):
            # The ellipsis isn't redundant
            newargs.append(ellipsis())

        endargs = []
        for i, s in enumerate(reversed(args[ellipsis_i+1:]), start=1):
            endargs.append(s.reduce(shape, axis=-i))

        newargs = newargs + endargs[::-1]

        return type(self)(*newargs)
