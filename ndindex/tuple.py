from numpy import broadcast, broadcast_to

from .ndindex import NDIndex, ndindex, asshape

class Tuple(NDIndex):
    """
    Represents a tuple of single-axis indices.

    Valid single axis indices are

    - :class:`Integer`
    - :class:`Slice`
    - :class:`ellipsis`
    - :class:`Newaxis`
    - :class:`IntegerArray`
    - :class:`BooleanArray`

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

    .. note::

       `Tuple` does *not* represent a tuple, but rather an *tuple index*. It
       does not have most methods that `tuple` has, and should not be used in
       non-indexing contexts. See the document on :ref:`type-confusion` for
       more details.

    """
    def _typecheck(self, *args):
        from .array import ArrayIndex
        from .ellipsis import ellipsis
        from .integerarray import IntegerArray
        from .booleanarray import BooleanArray

        newargs = []
        arrays = []
        for arg in args:
            newarg = ndindex(arg)
            if isinstance(newarg, Tuple):
                if len(args) == 1:
                    raise ValueError("tuples inside of tuple indices are not supported. Did you mean to call Tuple(*args) instead of Tuple(args)?")
                raise ValueError("tuples inside of tuple indices are not supported. If you meant to use a fancy index, use a list or array instead.")
            newargs.append(newarg)
            if isinstance(newarg, ArrayIndex):
                arrays.append(newarg)

        if newargs.count(ellipsis()) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if len(arrays) > 0 and isinstance(arrays[0], IntegerArray):
            if not all(isinstance(i, (IntegerArray, ellipsis)) for i in newargs):
                raise NotImplementedError("tuples mixing integer arrays with other index types are not yet supported")
            try:
                broadcast(*[i.raw for i in arrays])
            except ValueError as e:
                assert e.args == ("shape mismatch: objects cannot be broadcast to a single shape",)
                raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes %s" % ' '.join([str(i.shape) for i in arrays]))
        if len([i for i in newargs if isinstance(i, BooleanArray)]) > 0:
            raise NotImplementedError("tuples containing boolean arrays are not yet supported")

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
        tuple indices without an ellipsis always implicitly end in an
        ellipsis.

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
        Reduce a Tuple index on an array of shape `shape`

        A `Tuple` with a single argument is always reduced to that single
        argument (because `a[idx,]` is the same as `a[idx]`).

        >>> from ndindex import Tuple

        >>> Tuple(slice(2, 4)).reduce()
        Slice(2, 4, 1)

        If an explicit array shape is given, the result will either be
        `IndexError` if the index is invalid for the given shape, or an index
        that is as simple as possible:

        - All the elements of the tuple are recursively reduced.
        - Any axes that can be merged into an ellipsis are removed. This
          includes the implicit ellipsis at the end of a tuple that doesn't
          contain any explicit ellipses.
        - Ellipses that don't match any axes are removed.
        - An ellipsis at the end of the tuple is removed.
        - If the resulting Tuple would have a single argument, that argument
          is returned.

        >>> idx = Tuple(0, ..., slice(0, 3))
        >>> idx.reduce((5, 4))
        Tuple(0, slice(0, 3, 1))
        >>> idx.reduce((5, 3))
        Integer(0)

        >>> idx = Tuple(slice(0, 10), -3)
        >>> idx.reduce((5,))
        Traceback (most recent call last):
        ...
        IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        >>> idx.reduce((5, 2))
        Traceback (most recent call last):
        ...
        IndexError: index -3 is out of bounds for axis 1 with size 2

        Note
        ====

        ndindex presently does not distinguish between scalar objects and
        rank-0 arrays. It is possible for the original index to produce one
        and the reduced index to produce the other. In particular, the
        presence of a redundant ellipsis forces NumPy to return a rank-0 array
        instead of a scalar.

        >>> import numpy as np
        >>> a = np.array([0, 1])
        >>> Tuple(..., 1).reduce(a.shape)
        Integer(1)
        >>> a[..., 1]
        array(1)
        >>> a[1]
        1

        See https://github.com/Quansight/ndindex/issues/22.

        See Also
        ========

        .Tuple.expand
        .NDIndex.reduce
        .Slice.reduce
        .Integer.reduce
        .ellipsis.reduce
        .Newaxis.reduce
        .IntegerArray.reduce
        .BooleanArray.reduce

        """
        from .ellipsis import ellipsis
        from .slice import Slice

        args = self.args
        if ellipsis() not in args:
            return type(self)(*args, ellipsis()).reduce(shape)

        if shape is not None:
            # assert self.args.count(...) == 1
            n_newaxis = self.args.count(None)
            indexed_args = len(self.args) - n_newaxis - 1 # -1 for the ellipsis
            shape = asshape(shape, axis=indexed_args - 1)

        ellipsis_i = self.ellipsis_index

        preargs = []
        removable = shape is not None
        n_newaxis_before_ellipsis = args[:ellipsis_i].count(None)
        for i, s in enumerate(reversed(args[:ellipsis_i]), start=1):
            axis = ellipsis_i - i - n_newaxis_before_ellipsis
            if s == None:
                n_newaxis_before_ellipsis -= 1
                preargs.insert(0, s)
                removable = False
                continue
            reduced = s.reduce(shape, axis=axis)
            if (removable
                and isinstance(reduced, Slice)
                and reduced == Slice(0, shape[axis], 1)):
                continue
            else:
                removable = False
                preargs.insert(0, reduced)

        endargs = []
        removable = shape is not None
        for i, s in enumerate(args[ellipsis_i+1:]):
            axis = -len(args) + ellipsis_i + 1 + i
            axis += args[ellipsis_i+1:][i:].count(None)
            if shape is not None:
                # Make the axis positive so the error messages will match
                # numpy
                while axis < 0 and len(shape):
                    axis += len(shape)
            if s == None:
                endargs.append(s)
                removable = False
                continue
            reduced = s.reduce(shape, axis=axis)
            if (removable
                and isinstance(reduced, Slice)
                and reduced == Slice(0, shape[axis], 1)):
                continue
            else:
                removable = False
                endargs.append(reduced)

        if shape is None or (endargs and len(preargs) + len(endargs)
                             < len(shape) + args.count(None)):
            preargs = preargs + [...]

        newargs = preargs + endargs

        if newargs and newargs[-1] == ...:
            newargs = newargs[:-1]

        if len(newargs) == 1:
            return newargs[0]

        return type(self)(*newargs)


    def expand(self, shape):
        """
        Expand a Tuple index on an array of shape `shape`

        An expanded `Tuple` is one where the length of the .args is the same
        as the given shape plus the number of :any:`Newaxis` indices, and
        there are no ellipses.

        The result will either be `IndexError` if `self` is invalid for the
        given shape, or will be canonicalized so that

        - All the elements of the tuple are recursively reduced.

        - The length of the .args is equal to the length of the shape plus the
          number of :any:`Newaxis` indices in `self`.

        - The resulting Tuple has no ellipses. Axes that would be matched by
          an ellipsis or an implicit ellipsis at the end of the tuple are
          replaced by `Slice(0, n)`.

        - Any array indices in `self` are broadcast together. Note that
          broadcasting is done in a memory efficient way so that if the
          broadcasted shape is large it will not take up more memory than the
          original.

        >>> from ndindex import Tuple
        >>> idx = Tuple(slice(0, 10), ..., None, -3)

        >>> idx.expand((5, 3))
        Tuple(slice(0, 5, 1), None, 0)
        >>> idx.expand((1, 2, 3))
        Tuple(slice(0, 1, 1), slice(0, 2, 1), None, 0)

        >>> idx.expand((5,))
        Traceback (most recent call last):
        ...
        IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        >>> idx.expand((5, 2))
        Traceback (most recent call last):
        ...
        IndexError: index -3 is out of bounds for axis 1 with size 2

        See Also
        ========

        .Tuple.reduce
        .NDIndex.expand

        """
        from .array import ArrayIndex
        from .slice import Slice

        args = self.args
        if ... not in args:
            return type(self)(*args, ...).expand(shape)

        # Broadcast all array indices. Note that broadcastability checked in
        # the Tuple constructor, so this should not fail.
        arrays = [i.raw for i in self.args if isinstance(i, ArrayIndex)]
        broadcast_shape = broadcast(*arrays).shape
        newargs = []
        for s in args:
            if isinstance(s, ArrayIndex):
                # broadcast_to(x) gives a readonly view on x, which is also
                # readonly, so set _copy=False to avoid representing the full
                # broadcasted array in memory.
                s = type(s)(broadcast_to(s.raw, broadcast_shape), _copy=False)
            newargs.append(s)
        args = tuple(newargs)

        # assert self.args.count(...) == 1
        n_newaxis = self.args.count(None)
        indexed_args = len(self.args) - n_newaxis - 1 # -1 for the ellipsis
        shape = asshape(shape, axis=indexed_args - 1)

        ellipsis_i = self.ellipsis_index

        newargs = []
        n_newaxis_before_ellipsis = 0
        for i, s in enumerate(args[:ellipsis_i]):
            if s == None:
                n_newaxis_before_ellipsis += 1
                newargs.append(s)
            else:
                newargs.append(s.reduce(shape, axis=i - n_newaxis_before_ellipsis))

        newargs.extend([Slice(None).reduce(shape, axis=i + ellipsis_i - n_newaxis_before_ellipsis) for
                        i in range(len(shape) - len(args) + 1 + n_newaxis)])

        n_newaxis_after_ellipsis = 0
        endargs = []
        for i, s in enumerate(reversed(args[ellipsis_i+1:]), start=1):
            if s == None:
                n_newaxis_after_ellipsis += 1
                endargs.append(s)
            else:
                endargs.append(s.reduce(shape, axis=len(shape)-i+n_newaxis_after_ellipsis))

        newargs = newargs + endargs[::-1]

        return type(self)(*newargs)


    def newshape(self, shape):
        # The docstring for this method is on the NDIndex base class
        shape = asshape(shape)

        if self == Tuple():
            return shape

        # This will raise any IndexErrors
        self = self.expand(shape)

        ellipsis_i = self.ellipsis_index

        startshape = []
        n_newaxis = self.args.count(None)
        n_newaxis_before_ellipsis = 0
        for i, s in enumerate(self.args[:ellipsis_i]):
            if s == None:
                n_newaxis_before_ellipsis += 1
                startshape.append(1)
            else:
                startshape.extend(list(s.newshape(shape[i-n_newaxis_before_ellipsis])))

        n_newaxis_after_ellipsis = 0
        endshape = []
        for i, s in enumerate(reversed(self.args[ellipsis_i+1:]), start=1):
            if s == None:
                n_newaxis_after_ellipsis += 1
                endshape.append(1)
            else:
                endshape.extend(list(s.newshape(shape[len(shape)-i+n_newaxis_after_ellipsis])))

        if ... in self.args:
            midshape = list(shape[ellipsis_i-n_newaxis_before_ellipsis:len(shape)+ellipsis_i-len(self.args)+n_newaxis_after_ellipsis+1])
        else:
            midshape = list(shape[len(self.args) - n_newaxis:])

        newshape = startshape + midshape + endshape[::-1]

        return tuple(newshape)

    def as_subindex(self, index):
        from .ndindex import ndindex
        from .slice import Slice
        from .integer import Integer

        index = ndindex(index).reduce()

        if ... in self.args:
            raise NotImplementedError("Tuple.as_subindex() is not yet implemented for tuples with ellipses")

        if isinstance(index, Slice):
            if not self.args:
                if index.step < 0:
                    raise NotImplementedError("Tuple.as_subindex() is only implemented on slices with positive steps")
                return self

            first = self.args[0]
            return Tuple(first.as_subindex(index), *self.args[1:])
        if isinstance(index, Integer):
            index = Tuple(index)
        if isinstance(index, Tuple):
            new_args = []
            if any(isinstance(i, Slice) and i.step < 0 for i in index.args):
                    raise NotImplementedError("Tuple.as_subindex() is only implemented on slices with positive steps")
            if ... in index.args:
                raise NotImplementedError("Tuple.as_subindex() is not yet implemented for tuples with ellipses")
            for self_arg, index_arg in zip(self.args, index.args):
                subindex = self_arg.as_subindex(index_arg)
                if isinstance(subindex, Tuple):
                    continue
                new_args.append(subindex)
            return Tuple(*new_args, *self.args[min(len(self.args), len(index.args)):])
        raise NotImplementedError(f"Tuple.as_subindex() is not implemented for type '{type(index).__name__}")

    def isempty(self, shape=None):
        if shape is not None:
            return 0 in self.newshape(shape)

        return any(i.isempty() for i in self.args)
