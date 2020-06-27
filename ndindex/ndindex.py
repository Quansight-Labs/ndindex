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
    from . import Integer, Slice, Tuple, ellipsis

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

        See Also
        ========

        .Integer.reduce
        .Tuple.reduce
        .Slice.reduce
        .ellipsis.reduce

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

        An example usage of `as_subindex` is to split an index up into
        subindices of chunks of an array. For example, say a 1-D array `a` is
        chunked up into chunks of size `N`, so that `a[0:N]`, `a[N:2*N]`,
        `[2*N:3*N]`, etc. are stored separately. Then an index `a[i]` can be
        reindexed onto the chunks via `i.as_subindex(Slice(0, N))`,
        `i.as_subindex(Slice(N, 2*N))`, etc. See the example below.

        Examples
        ========

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
