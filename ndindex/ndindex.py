import inspect
import operator
import numbers
import warnings

from numpy import ndarray, asarray, integer, bool_, intp

def ndindex(obj):
    """
    Convert an object into an ndindex type

    Invalid indices will raise IndexError. Indices that are supported by NumPy
    but not yet supported by ndindex will raise NotImplementedError.

    >>> from ndindex import ndindex
    >>> ndindex(1)
    Integer(1)
    >>> ndindex(slice(0, 10))
    Slice(0, 10, None)
    """
    from . import Integer, Slice, Tuple, ellipsis, IntegerArray, BooleanArray

    if isinstance(obj, NDIndex):
        return obj

    if obj is None:
        raise NotImplementedError("newaxis is not yet implemented")

    # TODO: Replace this with calls to the IntegerArray() and BooleanArray()
    # constructors.
    if isinstance(obj, (list, ndarray, bool)):
        # Ignore deprecation warnings for things like [1, []]. These will be
        # filtered out anyway since they produce object arrays.
        with warnings.catch_warnings(record=True):
            a = asarray(obj)
            if isinstance(obj, list) and 0 in a.shape:
                a = a.astype(intp)
        if issubclass(a.dtype.type, integer):
            return IntegerArray(a)
        elif a.dtype == bool_:
            return BooleanArray(a)
        else:
            # Match the NumPy exceptions
            if isinstance(obj, ndarray):
                raise IndexError("arrays used as indices must be of integer (or boolean) type")
            else:
                raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")

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
        raise IndexError("Got ellipsis class. Did you mean to use the instance, ellipsis()?")
    if obj is Ellipsis:
        return ellipsis()

    raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")

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
    def __init__(self, *args, **kwargs):
        """
        This method should be called by subclasses (via super()) after type-checking
        """
        args = self._typecheck(*args, **kwargs)
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
        return f"{self.__class__.__name__}({', '.join(map(repr, self.args))})"

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        if not isinstance(other, NDIndex):
            try:
                other = ndindex(other)
            except (IndexError, NotImplementedError):
                return False

        def test_equal(a, b):
            """
            Check if a == b, allowing for numpy arrays
            """
            if not (isinstance(a, b.__class__)
                    or isinstance(b, a.__class__)):
                return False
            if isinstance(a, ndarray):
                return a.shape == b.shape and (a == b).all()
            if isinstance(a, tuple):
                return len(a) == len(b) and all(test_equal(i, j) for i, j in
                                                zip(a, b))
            if isinstance(a, NDIndex):
                return test_equal(a.args, b.args)

            return a == b

        return test_equal(self, other)

    def __hash__(self):
        return hash(self.args)

    # TODO: Make NDIndex an abstract base class
    @property
    def raw(self):
        """
        Return the equivalent of `self` that can be used as an index

        NumPy does not allow custom objects to be used as indices, with the
        exception of integer indices, so to use an ndindex object as an
        index, it is necessary to use `raw`.

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

        See Also
        ========

        .Integer.reduce
        .Tuple.reduce
        .Slice.reduce
        .ellipsis.reduce
        .IntegerArray.reduce
        .BooleanArray.reduce

        """
        # XXX: Should the default be raise NotImplementedError or return self?
        raise NotImplementedError

    def expand(self, shape):
        """
        Expand an index on an array of shape `shape`

        An expanded index is as explicit as possible. Unlike `reduce`, which
        tries to simplify an index and remove redundancies, `expand` typically
        makes an index larger.

        `expand` always returns a `Tuple` whose `.args` is the same length as
        `shape`. See :meth:`.Tuple.expand` for more details on the behavior of
        `expand`.

        >>> from ndindex import Slice
        >>> Slice(None).expand((2, 3))
        Tuple(slice(0, 2, 1), slice(0, 3, 1))

        See Also
        ========

        .Tuple.expand

        """
        from .tuple import Tuple

        return Tuple(self).expand(shape)

    def newshape(self, shape):
        """
        Returns the shape of `a[idx.raw]`, assuming `a` has shape `shape`.

        `shape` should be a tuple of ints, or an int, which is equivalent to a
        1-D shape.

        Raises IndexError if `self` would be out of shape for an array of
        shape `shape`.

        >>> from ndindex import Slice, Integer, Tuple
        >>> shape = (6, 7, 8)
        >>> Integer(1).newshape(shape)
        (7, 8)
        >>> Integer(10).newshape(shape)
        Traceback (most recent call last):
        ...
        IndexError: index 10 is out of bounds for axis 0 with size 6
        >>> Slice(2, 5).newshape(shape)
        (3, 7, 8)
        >>> Tuple(0, ..., Slice(1, 3)).newshape(shape)
        (7, 2)

        """
        raise NotImplementedError

    def as_subindex(self, index):
        """
        `i.as_subindex(j)` produces an index `k` such that `a[j][k]` gives all of
        the elements of `a[j]` that are also in `a[i]`.

        If `a[j]` is a subset of `a[i]`, then `a[j][k] == a[i]`. Otherwise,
        `a[j][k] == a[i & j]`, where `i & j` is the intersection of `i` and
        `j`, that is, the elements of `a` that are indexed by both `i` and
        `j`.

        For example, in the below diagram, `i` and `j` index a subset of the
        array `a`. `k = i.as_subindex(j)` is an index on `a[j]` that gives the
        subset of `a[j]` also included in `a[i]`::

             +------------ self ------------+
             |                              |
         ------------------- a -----------------------
                |                                 |
                +------------- index -------------+
                |                           |
                +- self.as_subindex(index) -+

        `i.as_subindex(j)` is currently only implemented when `j` is a slices
        with positive steps and nonnegative start and stop, or a Tuple of the
        same. To use it with slices with negative start or stop, call
        :meth:`reduce` with a shape first.

        `as_subindex` can be seen as the left-inverse of composition, that is,
        if `i = j[k]`, that is, `a[i] = a[j][k]`, then `k = i.as_subindex(j)`,
        so that `k "=" (j^-1)[i]` (this only works as a true inverse if
        `j` is a subset of `i`).

        Note that due to symmetry, `a[j][i.as_subindex(j)]` and
        `a[i][j.as_subindex(i)]` will give the same subarrays of `a`, which
        will be the array that includes the elements indexed by both `a[i]`
        and `a[j]`.

        `i.as_subindex(j)` may raise `ValueError` in the case that the indices
        `i` and `j` do not intersect at all.

        Examples
        ========

        An example usage of `as_subindex` is to split an index up into
        subindices of chunks of an array. For example, say a 1-D array `a` is
        chunked up into chunks of size `N`, so that `a[0:N]`, `a[N:2*N]`,
        `[2*N:3*N]`, etc. are stored separately. Then an index `a[i]` can be
        reindexed onto the chunks via `i.as_subindex(Slice(0, N))`,
        `i.as_subindex(Slice(N, 2*N))`, etc.

        >>> from ndindex import Slice
        >>> i = Slice(5, 15)
        >>> j1 = Slice(0, 10)
        >>> j2 = Slice(10, 20)
        >>> a = list(range(20))
        >>> a[i.raw]
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        >>> a[j1.raw]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> a[j2.raw]
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        >>> k1 = i.as_subindex(j1)
        >>> k1
        Slice(5, 10, 1)
        >>> k2 = i.as_subindex(j2)
        >>> k2
        Slice(0, 5, 1)
        >>> a[j1.raw][k1.raw]
        [5, 6, 7, 8, 9]
        >>> a[j2.raw][k2.raw]
        [10, 11, 12, 13, 14]

        """
        index = ndindex(index) # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__}.as_subindex({type(index).__name__}) isn't implemented yet")

    def isempty(self, shape=None):
        """
        Returns whether self always indexes an empty array

        An empty array is an array whose shape contains at least one 0. Note
        that scalars (arrays with shape `()`) are not considered empty.

        `shape` can be `None` (the default), or an array shape. If it is
        `None`, isempty() will return `True` when `self` is always empty for
        any array shape. However, if it gives `False`, it could still give an
        empty array for some array shapes, but not all. If you know the shape
        of the array that will be indexed, you can call `idx.isempty(shape)`
        first and the result will be correct for arrays of shape `shape`. If
        `shape` is given and `self` would raise an `IndexError` on an array of
        shape `shape`, `isempty()` also raises `IndexError`.

        >>> from ndindex import Tuple, Slice
        >>> Tuple(0, slice(0, 1)).isempty()
        False
        >>> Tuple(0, slice(0, 0)).isempty()
        True
        >>> Slice(5, 10).isempty()
        False
        >>> Slice(5, 10).isempty(4)
        True

        See Also
        ========
        ndindex.Slice.__len__

        """
        raise NotImplementedError

def asshape(shape, axis=None):
    """
    Cast `shape` as a valid NumPy shape.

    The input can be an integer `n`, which is equivalent to `(n,)`, or a tuple
    of integers.

    If the `axis` argument is provided, an `IndexError` is raised if it is out
    of bounds for the shape.

    The resulting shape is always a tuple of nonnegative integers.

    All ndindex functions that take a shape input should use::

        shape = asshape(shape)

    or::

        shape = asshape(shape, axis=axis)

    """
    from .integer import Integer
    from .tuple import Tuple
    if isinstance(shape, (Tuple, Integer)):
        raise TypeError("ndindex types are not meant to be used as a shape - "
                        "did you mean to use the built-in tuple type?")

    if isinstance(shape, numbers.Number):
        shape = (operator.index(shape),)

    try:
        l = len(shape)
    except TypeError:
        raise TypeError("expected sequence object with len >= 0 or a single integer")

    newshape = []
    # numpy uses __getitem__ rather than __iter__ to index into shape, so we
    # match that
    for i in range(l):
        # Raise TypeError if invalid
        newshape.append(operator.index(shape[i]))

        if shape[i] < 0:
            raise ValueError("unknown (negative) dimensions are not supported")

    if axis is not None:
        if len(newshape) <= axis:
            raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional, but {axis + 1} were indexed")

    return tuple(newshape)
