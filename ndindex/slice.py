import operator

from .ndindex import NDIndex

class default:
    """
    A default keyword argument value.

    Used as the default value for keyword arguments where `None` is also a
    meaningful value but not the default.

    """
    pass

class Slice(NDIndex):
    """
    Represents a slice on an axis of an nd-array.

    `Slice(x)` with one argument is equivalent to `Slice(None, x)`.

    `start` and `stop` can be any integer, or `None`. `step` can be any
    nonzero integer or `None`.

    `Slice(a, b)` is the same as the syntax `a:b` in an index and `Slice(a, b,
    c)` is the same as `a:b:c`. An argument being `None` is equivalent to the
    syntax where the item is omitted, for example, `Slice(None, None, k)` is
    the same as the syntax `::k`.

    `Slice` always has three arguments, and does not make any distinction
    between, for instance, `Slice(x, y)` and `Slice(x, y, None)`. This is
    because Python itself does not make the distinction between x:y and x:y:
    syntactically.

    See
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing
    for a description of the semantic meaning of slices on arrays.

    Slice has attributes `start`, `stop`, and `step` to access the
    corresponding attributes.

    >>> from ndindex import Slice
    >>> s = Slice(10)
    >>> s
    Slice(None, 10, None)
    >>> print(s.start)
    None
    >>> s.args
    (None, 10, None)
    >>> s.raw
    slice(None, 10, None)

    """
    def _typecheck(self, start, stop=default, step=None):
        if isinstance(start, Slice):
            return start.args
        if isinstance(start, slice):
            start, stop, step = start.start, start.stop, start.step

        if stop is default:
            start, stop = None, start

        if step == 0:
            raise ValueError("slice step cannot be zero")

        if start is not None:
            start = operator.index(start)
        if stop is not None:
            stop = operator.index(stop)
        if step is not None:
            step = operator.index(step)

        args = (start, stop, step)

        return args

    @property
    def raw(self):
        return slice(*self.args)

    @property
    def start(self):
        """
        The start value of the slice.

        Note that this may be an integer or None.
        """
        return self.args[0]

    @property
    def stop(self):
        """
        The stop of the slice.

        Note that this may be an integer or None.
        """
        return self.args[1]

    @property
    def step(self):
        """
        The step of the slice.

        This will be a nonzero integer.
        """
        return self.args[2]

    def __len__(self):
        """
        __len__ gives the maximum size of an axis sliced with self

        An actual array may produce a smaller size if it is smaller than the
        bounds of the slice. For instance, [0, 1, 2][2:4] only has 1 element
        but the maximum length of the slice 2:4 is 2.

        >>> from ndindex import Slice
        >>> [0, 1, 2][2:4]
        [2]
        >>> len(Slice(2, 4))
        2
        >>> [0, 1, 2, 3][2:4]
        [2, 3]

        If there is no such maximum, raises ValueError.

        >>> # From the second element to the end, which could have any size
        >>> len(Slice(1, None))
        Traceback (most recent call last):
        ...
        ValueError: Cannot determine max length of slice

        Note that the `Slice.reduce()` method returns a Slice that always has
        a correct `len` which doesn't raise ValueError.

        >>> Slice(2, 4).reduce(3)
        Slice(2, 3, 1)
        >>> len(_)
        1

        """
        start, stop, step = self.reduce().args
        error = ValueError("Cannot determine max length of slice")
        # We reuse the logic in range.__len__. However, it is only correct if
        # the slice doesn't use wrap around (see the comment in reduce()
        # below).
        if start is stop is None:
            raise error
        if step > 0:
            # start cannot be None
            if stop is None:
                if start >= 0:
                    # a[n:]. Extends to the end of the array.
                    raise error
                else:
                    # a[-n:]. From n from the end to the end. Same as
                    # range(-n, 0).
                    stop = 0
            elif start < 0 and stop >= 0:
                # a[-n:m] indexes from nth element from the end to the
                # m-1th element from the beginning.
                start, stop = 0, min(-start, stop)
            elif start >=0 and stop < 0:
                # a[n:-m]. The max length depends on the size of the array.
                raise error
        else:
            if start is None:
                if stop is None or stop >= 0:
                    # a[:m:-1] or a[::-1]. The max length depends on the size of
                    # the array
                    raise error
                else:
                    # a[:-m:-1]
                    start, stop = 0, -stop - 1
                    step = -step
            elif stop is None:
                if start >= 0:
                    # a[n::-1] (start != None by above). Same as range(n, -1, -1)
                    stop = -1
                else:
                    # a[-n::-1]. From n from the end to the beginning of the
                    # array backwards. The max length depends on the size of
                    # the array.
                    raise error
            elif start < 0 and stop >= 0:
                # a[-n:m:-1]. The max length depends on the size of the array
                raise error
            elif start >=0 and stop < 0:
                # a[n:-m:-1] indexes from the nth element backwards to the mth
                # element from the end.
                start, stop = 0, min(start+1, -stop - 1)
                step = -step

        return len(range(start, stop, step))

    def reduce(self, shape=None, axis=0):
        """
        `Slice.reduce` returns a slice where the start and stop are
        canonicalized for an array of the given shape, or for any shape if
        `shape` is `None` (the default).

        - If `shape` is `None`, the Slice is canonicalized so that

          - `start` and `stop` are not `None` when possible,
          - `step` is not `None`.

          Note that `start` and `stop` may be `None`, even after
          canonicalization with `reduce()` with no `shape`. This is because some
          slices are impossible to represent without `None` without making
          assumptions about the array shape. To get a slice where the `start`,
          `stop`, and `step` are always integers, use `reduce(shape)` with an
          explicit array shape.

          Note that `Slice` objects that index a single element are not
          canonicalized to `Integer`, because integer indices always remove an
          axis whereas slices keep the axis. Furthermore, slices cannot raise
          IndexError except on arrays with shape equal to `()`.

          >>> from ndindex import Slice
          >>> s = Slice(10)
          >>> s
          Slice(None, 10, None)
          >>> s.reduce()
          Slice(0, 10, 1)

        - If an explicit shape is given, the resulting object is always a
          `Slice` canonicalized so that

          - `start`, `stop`, and `step` are not `None`,
          - `start` is nonnegative.

          The `axis` argument can be used to specify an axis of the shape (by
          default, `axis=0`). For convenience, `shape` can be passed as an integer
          for a single dimension.

          After running `Slice.reduce(shape)` with an explicit shape, `len()`
          gives the true size of the axis for a sliced array of the given shape,
          and never raises ValueError.

          >>> from ndindex import Slice
          >>> s = Slice(1, 10)
          >>> s.reduce((3,))
          Slice(1, 3, 1)

          >>> s = Slice(2, None)
          >>> len(s)
          Traceback (most recent call last):
          ...
          ValueError: Cannot determine max length of slice
          >>> s.reduce((5,))
          Slice(2, 5, 1)
          >>> len(_)
          3

        """
        start, stop, step = self.args

        # Canonicalize with no shape

        if step is None:
            step = 1
        if start is None and step > 0:
            start = 0

        if start is not None and stop is not None:
            r = range(start, stop, step)
            # We can reuse some of the logic built-in to range(), but we have to
            # be careful. range() only acts like a slice if the 0 <= start <= stop (or
            # visa-versa for negative step). Otherwise, slices are different
            # because of wrap-around behavior. For example, range(-3, 1)
            # represents [-3, -2, -1, 0] whereas slice(-3, 1) represents the slice
            # of elements from the third to last to the first, which is either an
            # empty slice or a single element slice depending on the shape of the
            # axis.
            if len(r) == 0 and (
                    (step > 0 and start <= stop) or
                    (step < 0 and stop <= start)):
                start, stop, step = 0, 0, 1
            # This is not correct because a slice keeps the axis whereas an
            # integer index removes it.
            # if len(r) == 1:
            #     return Integer(r[0])

        if shape is None:
            return type(self)(start, stop, step)

        # Further canonicalize with an explicit array shape

        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) <= axis:
            raise IndexError("too many indices for array")

        size = shape[axis]

        # try:
        #     if len(self) == size:
        #         return self.__class__(None).reduce(shape, axis=axis)
        # except ValueError:
        #     pass
        if size == 0:
            start, stop, step = 0, 0, 1
        elif step > 0:
            # start cannot be None
            if start < 0:
                start = size + start
            if start < 0:
                start = 0

            if stop is None:
                stop = size
            elif stop < 0:
                stop = size + stop
                if stop < 0:
                    stop = 0
            else:
                stop = min(stop, size)
        else:
            if start is None:
                start = size - 1
            if stop is None:
                stop = -size - 1

            if start < 0:
                if start >= -size:
                    start = size + start
                else:
                    start, stop = 0, 0
            if start >= 0:
                start = min(size - 1, start)

            if -size <= stop < 0:
                stop += size
        return self.__class__(start, stop, step)
