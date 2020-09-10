from sympy.ntheory.modular import crt
from sympy import ilcm, Rational

from .ndindex import NDIndex, asshape, operator_index

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

    See :ref:`slices-docs` for a description of the semantic meaning of slices
    on arrays.

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
            start = operator_index(start)
        if stop is not None:
            stop = operator_index(stop)
        if step is not None:
            step = operator_index(step)

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
        `len()` gives the maximum size of an axis sliced with `self`.

        An actual array may produce a smaller size if it is smaller than the
        bounds of the slice. For instance, `[0, 1, 2][2:4]` only has 1 element
        but the maximum length of the slice `2:4` is 2.

        >>> from ndindex import Slice
        >>> [0, 1, 2][2:4]
        [2]
        >>> len(Slice(2, 4))
        2
        >>> [0, 1, 2, 3][2:4]
        [2, 3]

        If there is no such maximum, it raises `ValueError`.

        >>> # From the second element to the end, which could have any size
        >>> len(Slice(1, None))
        Traceback (most recent call last):
        ...
        ValueError: Cannot determine max length of slice

        The :meth:`Slice.reduce` method returns a Slice that always has a
        correct `len` which doesn't raise `ValueError`.

        >>> Slice(2, 4).reduce(3)
        Slice(2, 3, 1)
        >>> len(_)
        1

        Be aware that `len(Slice)` only gives the size of the axis being
        sliced. It does not say anything about the total shape of the array.
        In particular, the array may be empty after slicing if one of its
        dimensions is 0, but the other dimensions may be nonzero. To check if
        an array will empty after indexing, use :meth:`isempty`.

        See Also
        ========
        isempty

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

          - `start` is not `None`
          - `stop` is not `None` when possible,
          - `step` is not `None`.

          Note that `stop` may be `None`, even after canonicalization with
          `reduce()` with no `shape`. This is because some slices are
          impossible to represent without `None` without making assumptions
          about the array shape. To get a slice where the `start`, `stop`, and
          `step` are always integers, use `reduce(shape)` with an explicit
          array shape.

          Note that `Slice` objects that index a single element are not
          canonicalized to `Integer`, because integer indices always remove an
          axis whereas slices keep the axis. Furthermore, slices cannot raise
          `IndexError` except on arrays with shape equal to `()`.

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

        See Also
        ========

        .NDIndex.reduce
        .Tuple.reduce
        .Integer.reduce
        .ellipsis.reduce
        .Newaxis.reduce
        .IntegerArray.reduce
        .BooleanArray.reduce

        """
        start, stop, step = self.args

        # Canonicalize with no shape

        if step is None:
            step = 1
        if start is None:
            if step > 0:
                start = 0
            elif step < 0:
                start = -1

        if start == -1 and stop is None and step > 0:
            start, stop, step = (-1, -2, -1)
        elif start is not None and stop is not None:
            if start >= 0 and stop >= 0 or start < 0 and stop < 0:
                if step > 0:
                    if stop <= start:
                        start, stop, step = 0, 0, 1
                    elif start >= 0 and start + step >= stop:
                        # Indexes 1 element. Start has to be >= 0 because a
                        # negative start could be less than the size of the
                        # axis, in which case it will clip and the single
                        # element will be element 0. We can only do that
                        # reduction if we know the shape.

                        # Note that returning Integer here is wrong, because
                        # slices keep the axis and integers remove it.
                        stop, step = start + 1, 1
                    elif start < 0 and start + step > stop:
                        # The exception is this case where stop is already
                        # start + 1.
                        step = stop - start
                    if start >= 0:
                        stop -= (stop - start - 1) % step
                elif step < 0:
                    if stop >= start:
                        start, stop, step = 0, 0, 1
                    elif start < 0 and start + step <= stop:
                        if start < -1:
                            stop, step = start + 1, 1
                        else:
                            stop, step = start - 1, -1
                    elif stop == start - 1:
                        stop, step = start + 1, 1
                    if start < 0:
                        stop -= (stop - start + 1) % step
            elif start < 0 and stop == 0 and step > 0:
                start, stop, step = 0, 0, 1
            elif start < 0 and stop >= 0 and step >= min(-start, stop):
                step = min(-start, stop)
                if start == -1 or stop == 1:
                    # Can only index 0 or 1 elements. We can either pick a
                    # version with positive start and negative step, or
                    # negative start and positive step. We prefer the former
                    # as it matches what is done for reduce() with a shape
                    # (start is always nonnegative).
                    assert step == 1
                    start, stop, step = stop - 1, start - 1, -1
            elif step < 0 and stop < 0 and start < min(-step, -stop - 1):
                if stop == -1:
                    start, stop, step = 0, 0, 1
                else:
                    step = max(-start - 1, stop + 1)
            elif start >= 0 and stop == -1 and step < 0:
                start, stop, step = 0, 0, 1
        elif start is not None and stop is None:
            if start < 0 and step >= -start:
                step = -start
            elif step < 0:
                if start == 0:
                    start, stop, step = 0, 1, 1
                elif 0 <= start < -step:
                    step = -start - 1
        if shape is None:
            return type(self)(start, stop, step)

        # Further canonicalize with an explicit array shape

        shape = asshape(shape, axis=axis)
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

    def newshape(self, shape):
        # The docstring for this method is on the NDIndex base class
        shape = asshape(shape)

        idx = self.reduce(shape)

        # len() won't raise an error after reducing with a shape
        return (len(idx),) + shape[1:]

    # TODO: Better name?
    def as_subindex(self, index):
        # The docstring of this method is currently on NDindex.as_subindex, as
        # this is the only method that is actually implemented so far.

        from .ndindex import ndindex
        from .tuple import Tuple
        from .integer import Integer

        index = ndindex(index)

        s = self.reduce()
        index = index.reduce()

        if isinstance(index, Tuple):
            return Tuple(self).as_subindex(index)

        if isinstance(index, Integer):
            s = self.as_subindex(Slice(index.args[0], index.args[0] + 1))
            if s == Slice(0, 0, 1):
                # There is no index that we can return here. The intersection
                # of `self` and `index` is empty. Ideally we want to give an
                # index that gives an empty array, but we cannot make the
                # shape match. If a is dimension 1, then a[index] is dimension
                # 0, so a[index][slice(0, 0)] will not work. A possibility
                # would be to return False, which would add a length-0
                # dimension to the array. But
                #
                # 1. this isn't implemented yet, and
                # 2. a False can only add a length-0 dimension once, so it
                #    still wouldn't work in every case. For example,
                #    Tuple(slice(0), slice(0)).as_subindex((0, 0)) would need
                #    to return an index that replaces the first two
                #    dimensions with length-0 dimensions.
                raise ValueError(f"{self} and {index} do not intersect")
            assert len(s) == 1
            return Tuple()

        if not isinstance(index, Slice):
            raise NotImplementedError("Slice.as_subindex() is only implemented for tuples, integers and slices")

        if s.step < 0 or index.step < 0:
            raise NotImplementedError("Slice.as_subindex() is only implemented for slices with positive steps")

        # After reducing, start is not None when step > 0
        if index.stop is None or s.stop is None or s.start < 0 or index.start < 0 or s.stop < 0 or index.stop < 0:
            raise NotImplementedError("Slice.as_subindex() is only implemented for slices with nonnegative start and stop. Try calling reduce() with a shape first.")

        # Chinese Remainder Theorem. We are looking for a solution to
        #
        # x = s.start (mod s.step)
        # x = index.start (mod index.step)
        #
        # If crt() returns None, then there are no solutions (the slices do
        # not overlap).
        res = crt([s.step, index.step], [s.start, index.start])
        if res is None:
            return Slice(0, 0, 1)
        common, _ = res
        lcm = ilcm(s.step, index.step)
        start = max(s.start, index.start)

        def _smallest(x, a, m):
            """
            Gives the smallest integer >= x that equals a (mod m)

            Assumes x >= 0, m >= 1, and 0 <= a < m.
            """
            n = Rational(x - a, m).ceiling()
            return a + n*m

        # Get the smallest lcm multiple of common that is >= start
        start = _smallest(start, common, lcm)
        # Finally, we need to shift start so that it is relative to index
        start = (start - index.start)//index.step

        step = lcm//index.step # = s.step//igcd(s.step, index.step)

        stop = Rational((min(s.stop, index.stop) - index.start), index.step).ceiling()
        if stop < 0:
            stop = 0

        return Slice(start, stop, step).reduce()

    def isempty(self, shape=None):
        if shape is not None:
            return 0 in self.newshape(shape)

        try:
            l = len(self)
        except (TypeError, ValueError):
            return False
        return l == 0
