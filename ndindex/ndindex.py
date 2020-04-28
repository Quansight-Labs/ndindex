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

        """
        # XXX: Should the default be raise NotImplementedError or return self?
        raise NotImplementedError
