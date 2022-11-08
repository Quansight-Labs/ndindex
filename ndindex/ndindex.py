import sys
import inspect
import itertools
import numbers
import operator
import functools

newaxis = None

def ndindex(obj):
    """
    Convert an object into an ndindex type

    Invalid indices will raise `IndexError`.

    >>> from ndindex import ndindex
    >>> ndindex(1)
    Integer(1)
    >>> ndindex(slice(0, 10))
    Slice(0, 10, None)

    """
    if isinstance(obj, NDIndex):
        return obj

    if 'numpy' in sys.modules:
        from numpy import ndarray, bool_
    else: # pragma: no cover
        bool_ = bool
        ndarray = ()

    if isinstance(obj, (bool, bool_)):
        from . import BooleanArray
        return BooleanArray(obj)

    if isinstance(obj, (list, ndarray)):
        from . import IntegerArray, BooleanArray

        try:
            return IntegerArray(obj)
        except TypeError:
            pass
        try:
            return BooleanArray(obj)
        except TypeError:
            pass

        # Match the NumPy exceptions
        if isinstance(obj, ndarray):
            raise IndexError("arrays used as indices must be of integer (or boolean) type")
        else:
            raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")

    try:
        from . import Integer
        # If operator.index() works, use that
        return Integer(obj)
    except TypeError:
        pass

    if isinstance(obj, slice):
        from . import Slice
        return Slice(obj)

    if isinstance(obj, tuple):
        from . import Tuple
        return Tuple(*obj)

    from . import ellipsis

    if obj == ellipsis:
        raise IndexError("Got ellipsis class. Did you mean to use the instance, ellipsis()?")
    if obj is Ellipsis:
        return ellipsis()

    if obj == newaxis:
        from . import Newaxis
        return Newaxis()

    raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

class ImmutableObject:
    """
    Base class for immutable objects.

    Subclasses of this class are immutable objects. They all have the `.args`
    attribute, which gives the full necessary data to recreate the class, via,

    .. code:: python

       type(obj)(*obj.args) == obj

    Note: subclasses that specifically represent indices should subclass
    :class:`NDIndex` instead.

    All classes that subclass `ImmutableObject` should define the `_typecheck`
    method. `_typecheck(self, *args)` should do type checking and basic type
    canonicalization, and either return a tuple of the new arguments for the
    class or raise an exception. Type checking means it should raise
    exceptions for input types that are never semantically meaningful for
    numpy arrays, for example, floating point indices, using the same
    exceptions as numpy where possible. Basic type canonicalization means, for
    instance, converting integers into `int` using `operator.index()`. All
    other canonicalization should be done in the `reduce()` method. The
    `ImmutableObject` base constructor will automatically set `.args` to the
    arguments returned by this method. Classes should always be able to
    recreate themselves with `.args`, i.e., `type(obj)(*obj.args) == obj`
    should always hold.

    See Also
    ========

    NDIndex

    """
    __slots__ = ('args',)

    def __init__(self, *args, **kwargs):
        """
        This method should be called by subclasses (via super()) after type-checking
        """
        args = self._typecheck(*args, **kwargs)
        self.args = args
        """
        `idx.args` contains the arguments needed to create `idx`.

        For an ndindex object `idx`, `idx.args` is always a tuple such that

        .. code:: python

           type(idx)(*idx.args) == idx

        For :class:`Tuple` indices, the elements of `.args` are themselves
        ndindex types. For other types, `.args` contains raw Python types.
        Note that `.args` contains NumPy arrays for :class:`~.IntegerArray`
        and :class:`~.BooleanArray` types, so one should always do equality
        testing or hashing on the ndindex type itself, not its `.args`.

        """
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
        if not isinstance(other, ImmutableObject):
            try:
                other = self.__class__(other)
            except TypeError:
                return False

        return self.args == other.args

    def __hash__(self): # pragma: no cover
        # Note: subclasses where .args is not hashable should redefine
        # __hash__
        return hash(self.args)

class NDIndex(ImmutableObject):
    """
    Represents an index into an nd-array (i.e., a numpy array).

    This is a base class for all ndindex types. All types that subclass this
    class should redefine the following methods

    - `_typecheck(self, *args)`. See the docstring of
      :class:`ImmutableObject`.

    - `raw` (a **@property** method) should return the raw index that can be
      passed as an index to a numpy array.

    In addition other methods should be defined as necessary.

    - `__len__` should return the largest possible shape of an axis sliced by
      the index (for single-axis indices), or raise ValueError if no such
      maximum exists.

    - `reduce(shape=None)` should reduce an index to an equivalent form for
      arrays of shape `shape`, or raise an `IndexError`. The error messages
      should match numpy as much as possible. The class of the equivalent
      index may be different. If `shape` is `None`, it should return a
      canonical form that is equivalent for all array shapes (assuming no
      IndexErrors).

    The methods `__init__` and `__eq__` should *not* be overridden. Equality
    (and hashability) on `NDIndex` subclasses is determined by equality of
    types and `.args`. Equivalent indices should not attempt to redefine
    equality. Rather they should define canonicalization via `reduce()`.
    `__hash__` is defined so that the hash matches the hash of `.raw`. If
    `.raw` is unhashable, `__hash__` should be overridden to use
    `hash(self.args)`.

    See Also
    ========

    ImmutableObject

    """
    __slots__ = ()

    # TODO: Make NDIndex and ImmutableObject abstract base classes
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

    # This is still here as a fallback implementation, but it isn't actually
    # used by any present subclasses, because it is faster to implement __eq__
    # on each class specifically.
    def __eq__(self, other): # pragma: no cover
        if not isinstance(other, NDIndex):
            try:
                other = ndindex(other)
                return self == other
            except IndexError:
                return False
        return super().__eq__(other)

    def __hash__(self):
        # Make the hash match the raw hash when the raw type is hashable.
        # Note: subclasses where .raw is not hashable should define __hash__
        # as hash(self.args)
        return hash(self.raw)

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
        .Newaxis.reduce
        .IntegerArray.reduce
        .BooleanArray.reduce

        """
        # XXX: Should the default be raise NotImplementedError or return self?
        raise NotImplementedError

    def expand(self, shape):
        r"""
        Expand a Tuple index on an array of shape `shape`

        An expanded index is as explicit as possible. Unlike :meth:`reduce
        <ndindex.ndindex.NDIndex.reduce>`, which tries to simplify an index
        and remove redundancies, `expand()` typically makes an index larger.

        If `self` is invalid for the given shape, an `IndexError` is raised.
        Otherwise, the returned index satisfies the following:

        - It is always a :class:`Tuple`.

        - All the elements of the :class:`Tuple` are recursively :any:`reduced
          <ndindex.ndindex.NDIndex.reduce>`.

        - The length of the `.args` is equal to the length of the shape plus
          the number of :class:`~.Newaxis` indices in `self` plus 1 if there
          is a scalar :class:`~.BooleanArray` (`True` or `False`).

        - The resulting :class:`Tuple` has no :class:`ellipses
          <ndindex.ellipsis>`. If there are axes that would be matched by an
          ellipsis or an implicit ellipsis at the end of the tuple, `Slice(0,
          n, 1)` indices are inserted, where `n` is the corresponding axis of
          the `shape`.

        - Any array indices in `self` are broadcast together. If `self`
          contains array indices (:class:`~.IntegerArray` or
          :class:`~.BooleanArray`), then any :class:`~.Integer` indices are
          converted into :class:`~.IntegerArray` indices of shape `()` and
          broadcast. Note that broadcasting is done in a memory efficient way
          so that even if the broadcasted shape is large it will not take up
          more memory than the original.

        - Scalar :class:`~.BooleanArray` arguments (`True` or `False`) are
          combined into a single term (the same as with :meth:`.Tuple.reduce`).

        - Non-scalar :class:`~.BooleanArray`\ s are all converted into
          equivalent :class:`~.IntegerArray`\ s via `nonzero()` and broadcast.

        >>> from ndindex import Tuple, Slice
        >>> Slice(None).expand((2, 3))
        Tuple(slice(0, 2, 1), slice(0, 3, 1))

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

        >>> idx = Tuple(..., [0, 1], -1)
        >>> idx.expand((1, 2, 3))
        Tuple(slice(0, 1, 1), [0, 1], [2, 2])

        See Also
        ========

        .Tuple.reduce
        broadcast_arrays

        """
        from .tuple import Tuple

        return Tuple(self).expand(shape)

    def newshape(self, shape):
        """
        Returns the shape of `a[idx.raw]`, assuming `a` has shape `shape`.

        `shape` should be a tuple of ints, or an int, which is equivalent to a
        1-D shape.

        Raises `IndexError` if `self` would be out of shape for an array of
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

        `i.as_subindex(j)` is currently only implemented when `j` is a slice
        with positive steps and nonnegative start and stop, or a Tuple of the
        same. To use it with slices with negative start or stop, call
        :meth:`reduce` with a shape first.

        `as_subindex` can be seen as the left-inverse of composition, that is,
        if `a[i] = a[j][k]`, then `k = i.as_subindex(j)`, so that `k "="
        (j^-1)[i]` (this only works as a true inverse if `j` is a subset of
        `i`).

        Note that due to symmetry, `a[j][i.as_subindex(j)]` and
        `a[i][j.as_subindex(i)]` will give the same subarrays of `a`, which
        will be the array of elements indexed by both `a[i]` and `a[j]`.

        `i.as_subindex(j)` may raise `ValueError` in the case that the indices
        `i` and `j` do not intersect at all.

        Examples
        ========

        An example usage of `as_subindex` is to split an index up into
        subindices of chunks of an array. For example, say a 1-D array
        `a` is chunked up into chunks of size `N`, so that `a[0:N]`,
        `a[N:2*N]`, `[2*N:3*N]`, etc. are stored separately. Then an index
        `a[i]` can be reindexed onto the chunks via `i.as_subindex(Slice(0,
        N))`, `i.as_subindex(Slice(N, 2*N))`, etc.

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

        See Also
        ========

        ndindex.ChunkSize.as_subchunks:
            a high-level iterator that efficiently gives only those chunks
            that intersect with a given index
        ndindex.ChunkSize.num_subchunks

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

    def broadcast_arrays(self):
        """
        Broadcast all the array indices in self to a common shape and convert
        boolean array indices into integer array indices.

        The resulting index is equivalent in all contexts where the original
        index is allowed. However, it is possible for the original index to
        give an IndexError but for the new index to not, since integer array
        indices have less stringent shape requirements than boolean array
        indices. There are also some instances for empty indices
        (:any:`isempty` is True) where bounds would be checked before
        broadcasting but not after.

        Any :class:`~.BooleanArray` indices are converted to
        :class:`~.IntegerArray` indices. Furthermore, if there are
        :class:`~.BooleanArray` or :class:`~.IntegerArray` indices, then any
        :class:`~.Integer` indices are also converted into scalar
        :class:`~.IntegerArray` indices and broadcast. Furthermore, if there are
        multiple boolean scalar indices (`True` or `False`), they are combined
        into a single one.

        Note that array broadcastability is checked in the :class:`Tuple`
        constructor, so this method will not raise any exceptions.

        This is part of what is performed by :any:`expand`, but unlike
        :any:`expand`, this method does not do any other manipulations, and it
        does not require a shape.

        >>> from ndindex import Tuple
        >>> idx = Tuple([[False], [True], [True]], [[4], [5], [5]], -1)
        >>> print(idx.broadcast_arrays())
        Tuple(IntegerArray([[1 2] [1 2] [1 2]]),
              IntegerArray([[0 0] [0 0] [0 0]]),
              IntegerArray([[4 4] [5 5] [5 5]]),
              IntegerArray([[-1 -1] [-1 -1] [-1 -1]]))

        See Also
        ========

        expand

        """
        return self


# TODO: Use this in other places in the code that check broadcast compatibility.
class BroadcastError(ValueError):
    """
    Exception raised by :func:`iter_indices()` when the input shapes are not
    broadcast compatible.

    This is used instead of the NumPy exception of the same name so that
    `iter_indices` does not need to depend on NumPy.
    """

class AxisError(ValueError, IndexError):
    """
    Exception raised by :func:`iter_indices()` when the `skip_axes` argument
    is out of bounds.

    This is used instead of the NumPy exception of the same name so that
    `iter_indices` does not need to depend on NumPy.

    """
    __slots__ = ("axis", "ndim")

    def __init__(self, axis, ndim):
        self.axis = axis
        self.ndim = ndim

    def __str__(self):
        return f"axis {self.axis} is out of bounds for array of dimension {self.ndim}"

def broadcast_shapes(*shapes):
    """
    Broadcast the input shapes `shapes` to a single shape.

    This is the same as :py:func:`np.broadcast_shapes()
    <numpy.broadcast_shapes>`. It is included as a separate helper function
    because `np.broadcast_shapes()` is on available in NumPy 1.20 or newer, and
    so that ndindex functions that use this function can do without requiring
    NumPy to be installed.

    """

    def _broadcast_shapes(shape1, shape2):
        """Broadcasts `shape1` and `shape2`"""
        N1 = len(shape1)
        N2 = len(shape2)
        N = max(N1, N2)
        shape = [None for _ in range(N)]
        i = N - 1
        while i >= 0:
            n1 = N1 - N + i
            if N1 - N + i >= 0:
                d1 = shape1[n1]
            else:
                d1 = 1
            n2 = N2 - N + i
            if N2 - N + i >= 0:
                d2 = shape2[n2]
            else:
                d2 = 1

            if d1 == 1:
                shape[i] = d2
            elif d2 == 1:
                shape[i] = d1
            elif d1 == d2:
                shape[i] = d1
            else:
                # TODO: Build an error message that matches NumPy
                raise BroadcastError("shape mismatch: objects cannot be broadcast to a single shape.")

            i = i - 1

        return tuple(shape)

    return functools.reduce(_broadcast_shapes, shapes, ())

def iter_indices(*shapes, skip_axes=(), _debug=False):
    """
    Iterate indices for every element of an arrays of shape `shapes`.

    Each shape in `shapes` should be a shape tuple, which are broadcast
    compatible. Each iteration step will produce a tuple of indices, one for
    each shape, which would correspond to the same elements if the arrays of
    the given shapes were first broadcast together.

    This is a generalization of the NumPy :py:class:`np.ndindex()
    <numpy.ndindex>` function (which otherwise has no relation).
    `np.ndindex()` only iterates indices for a single shape, whereas
    `iter_indices()` supports generating indices for multiple broadcast
    compatible shapes at once. This is equivalent to first broadcasting the
    arrays then generating indices for the single broadcasted shape.

    Additionally, this function supports the ability to skip axes of the
    shapes using `skip_axes`. These axes will be fully sliced in each index.
    The remaining axes will be indexed one element at a time with integer
    indices.

    `skip_axes` should be a tuple of axes to skip. It can use negative
    integers, e.g., `skip_axes=(-1,)` will skip the last axis. The order of
    the axes in `skip_axes` does not matter, but it should not contain
    duplicate axes. The axes in `skip_axes` refer to the final broadcasted
    shape of `shapes`. For example, `iter_indices((3,), (1, 2, 3),
    skip_axes=(0,))` will skip the first axis, and only applies to the second
    shape, since the first shape corresponds to axis `2` of the final
    broadcasted shape `(1, 2, 3)`

    For example, suppose `a` is an array with shape `(3, 2, 4, 4)`, which we
    wish to think of as a `(3, 2)` stack of 4 x 4 matrices. We can generate an
    iterator for each matrix in the "stack" with `iter_indices((3, 2, 4, 4),
    skip_axes=(-1, -2))`:

    >>> from ndindex import iter_indices
    >>> for idx in iter_indices((3, 2, 4, 4), skip_axes=(-1, -2)):
    ...     print(idx)
    (Tuple(0, 0, slice(None, None, None), slice(None, None, None)),)
    (Tuple(0, 1, slice(None, None, None), slice(None, None, None)),)
    (Tuple(1, 0, slice(None, None, None), slice(None, None, None)),)
    (Tuple(1, 1, slice(None, None, None), slice(None, None, None)),)
    (Tuple(2, 0, slice(None, None, None), slice(None, None, None)),)
    (Tuple(2, 1, slice(None, None, None), slice(None, None, None)),)

    Note that the iterates of `iter_indices` are always a tuple, even if only
    a single shape is provided (one could instead use `for idx, in
    iter_indices(...)` above).

    As another example, say `a` is shape `(1, 3)` and `b` is shape `(2, 1)`,
    and we want to generate indices for every value of the broadcasted
    operation `a + b`. We can do this by using `a[idx1.raw] + b[idx2.raw]` for every
    `idx1` and `idx2` as below:

    >>> import numpy as np
    >>> a = np.arange(3).reshape((1, 3))
    >>> b = np.arange(100, 111, 10).reshape((2, 1))
    >>> a
    array([[0, 1, 2]])
    >>> b
    array([[100],
           [110]])
    >>> for idx1, idx2 in iter_indices((1, 3), (2, 1)): # doctest: +SKIP37
    ...     print(f"{idx1 = }; {idx2 = }; {(a[idx1.raw], b[idx2.raw]) = }")
    idx1 = Tuple(0, 0); idx2 = Tuple(0, 0); (a[idx1.raw], b[idx2.raw]) = (0, 100)
    idx1 = Tuple(0, 1); idx2 = Tuple(0, 0); (a[idx1.raw], b[idx2.raw]) = (1, 100)
    idx1 = Tuple(0, 2); idx2 = Tuple(0, 0); (a[idx1.raw], b[idx2.raw]) = (2, 100)
    idx1 = Tuple(0, 0); idx2 = Tuple(1, 0); (a[idx1.raw], b[idx2.raw]) = (0, 110)
    idx1 = Tuple(0, 1); idx2 = Tuple(1, 0); (a[idx1.raw], b[idx2.raw]) = (1, 110)
    idx1 = Tuple(0, 2); idx2 = Tuple(1, 0); (a[idx1.raw], b[idx2.raw]) = (2, 110)
    >>> a + b
    array([[100, 101, 102],
           [110, 111, 112]])

    To include an index into the final broadcasted array, you can simply
    include the final broadcasted shape as one of the shapes (the NumPy
    function :func:`np.broadcast_shapes() <numpy:numpy.broadcast_shapes>` is
    useful here).

    >>> np.broadcast_shapes((1, 3), (2, 1))
    (2, 3)
    >>> for idx1, idx2, broadcasted_idx in iter_indices((1, 3), (2, 1), (2, 3)):
    ...     print(broadcasted_idx)
    Tuple(0, 0)
    Tuple(0, 1)
    Tuple(0, 2)
    Tuple(1, 0)
    Tuple(1, 1)
    Tuple(1, 2)

    """
    if not shapes:
        yield ()
        return

    shapes = [asshape(shape) for shape in shapes]
    ndim = len(max(shapes, key=len))

    if isinstance(skip_axes, int):
        skip_axes = (skip_axes,)
    _skip_axes = []
    for a in skip_axes:
        try:
            a = ndindex(a).reduce(ndim).args[0]
        except IndexError:
            raise AxisError(a, ndim)
        if a in _skip_axes:
            raise ValueError("skip_axes should not contain duplicate axes")
        _skip_axes.append(a)

    _shapes = [(1,)*(ndim - len(shape)) + shape for shape in shapes]
    iters = [[] for i in range(len(shapes))]
    broadcasted_shape = broadcast_shapes(*shapes)

    for i in range(-1, -ndim-1, -1):
        for it, shape, _shape in zip(iters, shapes, _shapes):
            if -i > len(shape):
                for j in range(len(it)):
                    if broadcasted_shape[i] not in [0, 1]:
                        it[j] = ncycles(it[j], broadcasted_shape[i])
                    break
            elif ndim + i in _skip_axes:
                it.insert(0, [slice(None)])
            else:
                if broadcasted_shape[i] != 1 and shape[i] == 1:
                    it.insert(0, ncycles(range(shape[i]), broadcasted_shape[i]))
                else:
                    it.insert(0, range(shape[i]))

    if _debug: # pragma: no cover
        print(iters)
        # Use this instead when we drop Python 3.7 support
        # print(f"{iters = }")
    for idxes in itertools.zip_longest(*[itertools.product(*i) for i in
                                         iters], fillvalue=()):
        yield tuple(ndindex(idx) for idx in idxes)

class ncycles:
    """
    Iterate `iterable` repeated `n` times.

    This is based on a recipe from the `Python itertools docs
    <https://docs.python.org/3/library/itertools.html#itertools-recipes>`_,
    but improved to give a repr, and to denest when it can. This makes
    debugging :func:`~.iter_indices` easier.

    >>> from ndindex.ndindex import ncycles
    >>> ncycles(range(3), 2)
    ncycles(range(0, 3), 2)
    >>> list(_)
    [0, 1, 2, 0, 1, 2]
    >>> ncycles(ncycles(range(3), 3), 2)
    ncycles(range(0, 3), 6)

    """
    def __new__(cls, iterable, n):
        if n == 1:
            return iterable
        return object.__new__(cls)

    def __init__(self, iterable, n):
        if isinstance(iterable, ncycles):
            self.iterable = iterable.iterable
            self.n = iterable.n*n
        else:
            self.iterable = iterable
            self.n = n

    def __repr__(self):
        return f"ncycles({self.iterable!r}, {self.n!r})"

    def __iter__(self):
        return itertools.chain.from_iterable(itertools.repeat(tuple(self.iterable), self.n))

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
        shape = (operator_index(shape),)

    try:
        l = len(shape)
    except TypeError:
        raise TypeError("expected sequence object with len >= 0 or a single integer")

    newshape = []
    # numpy uses __getitem__ rather than __iter__ to index into shape, so we
    # match that
    for i in range(l):
        # Raise TypeError if invalid
        newshape.append(operator_index(shape[i]))

        if shape[i] < 0:
            raise ValueError("unknown (negative) dimensions are not supported")

    if axis is not None:
        if len(newshape) <= axis:
            raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional, but {axis + 1} were indexed")

    return tuple(newshape)

def operator_index(idx):
    """
    Convert `idx` into an integer index using `__index__()` or raise
    `TypeError`.

    This is the same as `operator.index()` except it disallows boolean types.

    This is a slight break in NumPy compatibility, as NumPy allows bools in
    some contexts where `__index__()` is used, for instance, in slices. It
    does disallow it in others, such as in shapes. The main motivation for
    disallowing bools entirely is 1) `numpy.bool_.__index__()` is deprecated
    (currently it matches the built-in `bool.__index__()` and returns the
    object unchanged, but prints a deprecation warning), and 2) for raw
    indices, booleans and `0`/`1` are completely different, i.e., `a[True]` is
    *not* the same as `a[1]`.

    >>> from ndindex.ndindex import operator_index
    >>> operator_index(1)
    1
    >>> operator_index(1.0)
    Traceback (most recent call last):
    ...
    TypeError: 'float' object cannot be interpreted as an integer
    >>> operator_index(True)
    Traceback (most recent call last):
    ...
    TypeError: 'bool' object cannot be interpreted as an integer

    """
    if isinstance(idx, bool):
        raise TypeError("'bool' object cannot be interpreted as an integer")
    if 'numpy' in sys.modules and isinstance(idx, sys.modules['numpy'].bool_):
        raise TypeError("'np.bool_' object cannot be interpreted as an integer")
    return operator.index(idx)
