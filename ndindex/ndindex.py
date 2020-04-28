import operator
import inspect

from numpy import ndarray

def ndindex(obj):
    """
    Convert an object into an ndindex type

    >>> from ndindex import ndindex
    >>> ndindex(1)
    Integer(1)
    >>> ndindex(slice(0, 10))
    Slice(0, 10, None)
    """

    if isinstance(obj, NDIndex):
        return obj

    try:
        # If operator.index() works, use that
        return Integer(obj)
    except TypeError:
        pass

    if isinstance(obj, slice):
        return Slice(obj)

    if isinstance(obj, tuple):
        return Tuple(*obj)

    if obj == ellipsis:
        raise TypeError("Got ellipsis class. Did you mean to use the instance, ellipsis()?")
    if obj is Ellipsis:
        return ellipsis()

    if isinstance(obj, ndarray):
        raise NotImplementedError("array indices are not yet supported")

    raise TypeError(f"Don't know how to convert object of type {type(obj)} to an ndindex object")

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

class NDIndex:
    """
    Represents an index into an nd-array (i.e., a numpy array).

    This is a base class for all ndindex types. All types that subclass this
    class should redefine the following methods

    - `_typecheck(self, *args)` should do type checking and basic type
      canonicalization, and either return a tuple of the new arguments for the
      class or raise an exception. Type checking means it should raise
      exceptions for input types that are never semantically meaningful for
      numpy arrays, for example, floating point indices, using the same
      exceptions as numpy where possible. Basic type canonicalization means,
      for instance, converting integers into `int` using `operator.index()`.
      All other canonicalization should be done in the `reduce()` method. The
      `NDIndex` base constructor will automatically set `.args` to the
      arguments returned by this method. Classes should always be able to
      recreate themselves with `.args`, i.e., `type(idx)(*idx.args) == idx`
      should always hold.

    - `raw` (a **@property** method) should return the raw index that can be
      passed as an index to a numpy array.

    In addition other methods should be defined as necessary.

    - `__len__` should return the largest possible shape of an axis sliced by
      the index (for single-axis indices), or raise ValueError if no such
      maximum exists.

    - `reduce(shape=None)` should reduce an index to an equivalent form for
      arrays of shape `shape`, or raise an IndexError. The error messages
      should match numpy as much as possible. The class of the equivalent
      index may be different. If `shape` is `None`, it should return a
      canonical form that is equivalent for all array shapes (assuming no
      IndexErrors).

    The methods `__init__`, `__eq__`, and `__hash__` should *not* be
    overridden. Equality (and hashability) on `NDIndex` subclasses is
    determined by equality of types and `.args`. Equivalent indices should not
    attempt to redefine equality. Rather they should define canonicalization
    via `reduce()`.

    """
    def __init__(self, *args):
        """
        This method should be called by subclasses (via super()) after type-checking
        """
        args = self._typecheck(*args)
        self.args = args

    @classproperty
    def __signature__(self):
        """
        Allow Python 3's inspect.signature to give a useful signature for
        NDIndex subclasses.
        """
        sig = inspect.signature(self._typecheck)
        d = dict(sig.parameters)
        d.pop('self')
        return inspect.Signature(d.values())

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        if not isinstance(other, NDIndex):
            try:
                other = ndindex(other)
            except (TypeError, NotImplementedError):
                return False

        return ((isinstance(other, self.__class__)
                 or isinstance(self, other.__class__))
                and self.args == other.args)

    def __hash__(self):
        return hash(self.args)

    # TODO: Make NDIndex an abstract base class
    @property
    def raw(self):
        """
        Return the equivalent of `self` that can be used as an index

        NumPy does not allow custom objects to be used as indices, with the
        exception of integer indices, so to use an NDIndex object as an index,
        it is necessary to use `raw`.

        >>> from ndindex import Slice
        >>> import numpy as np
        >>> a = np.arange(5)
        >>> s = Slice(2, 4)
        >>> a[s]
        Traceback (most recent call last):
        ...
        IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
        >>> a[s.raw]
        array([2, 3])

        """
        raise NotImplementedError

    def reduce(self, shape=None):
        """
        Simplify an index given that it will be applied to an array of a given shape.

        If `shape` is None (the default), the index will be canonicalized as
        much as possible while still staying equivalent for all array shapes
        that it does not raise IndexError for.

        Either returns a new index type, which is equivalent on arrays of
        shape `shape`, or raises IndexError if the index would give an index
        error (for instance, out of bounds integer index or too many indices
        for array).

        >>> from ndindex import Slice, Integer
        >>> Slice(0, 10).reduce((5,))
        Slice(0, 5, 1)
        >>> Integer(10).reduce((5,))
        Traceback (most recent call last):
        ...
        IndexError: index 10 is out of bounds for axis 0 with size 5

        For single axis indices such as Slice and Tuple, `reduce` takes an
        optional `axis` argument to specify the axis, defaulting to 0.

        """
        # XXX: Should the default be raise NotImplementedError or return self?
        raise NotImplementedError

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
    unlike `Ellipsis`, `ellipsis()` is not singletonized.

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

    >>> from ndindex import Tuple
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
        newargs = []
        for arg in args:
            newarg = ndindex(arg)
            if isinstance(newarg, Tuple):
                raise NotImplementedError("tuples of tuples are not yet supported")
            newargs.append(newarg)

        if newargs.count(ellipsis()) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        return tuple(newargs)

    @property
    def has_ellipsis(self):
        """
        Returns True if self has an ellipsis
        """
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

        >>> Tuple(Slice(2, 4)).reduce()
        Slice(2, 4, 1)

        If an explicit array shape is given, The result will either be
        IndexError if the index is invalid for the given shape, or Tuple where
        the entries are recursively reduced.

        >>> from ndindex import Tuple, Integer, Slice
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
        Tuple(Slice(0, 5, 1), Integer(0))

        """
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
