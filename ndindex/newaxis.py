from .ndindex import NDIndex


class Newaxis(NDIndex):
    """
    Represents a `np.newaxis` (i.e., `None`) index.

    Newaxis adds a shape 1 dimension to the array. If a Newaxis is inside of a
    tuple index, it adds a shape 1 dimension at that location in the index.

    For example, if `a` has shape `(2, 3)`, then `a[newaxis]` has shape `(1,
    2, 3)`, `a[:, newaxis]` has shape (2, 1, 3)`, and so on.

    >>> from ndindex import Newaxis
    >>> from numpy import arange
    >>> a = arange(0,6).reshape(2,3)
    >>> a[Newaxis().raw].shape
    (1, 2, 3)
    >>> a[:, Newaxis().raw, :].shape
    (2, 1, 3)

    Using `Newaxis().raw` as an index is equivalent to using `numpy.newaxis`.

    **Note:** Unlike the NumPy `newaxis`, `Newaxis` is the type, not the
    object (the name is lowercase to avoid conflicting with the NumPy type).
    Use `Newaxis()`, `ndindex(np.newaxis)`, or `ndindex(None)` to create the
    object. In most ndindex contexts, `np.newaxis` or `None` can be used
    instead of `Newaxis()`, for instance, when creating a `Tuple` object. Also
    unlike `None`, `Newaxis()` is not singletonized, so you should not use
    `is` to compare it. See the document on :ref:`type-confusion` for more
    details.

    """
    def _typecheck(self):
        return ()

    def reduce(self, shape=None):
        """
        Reduce Newaxis index

        For any shape, `Newaxis().reduce(shape)` returns and index equivalent
        to `Newaxis()`.
        """
        return self

    @property
    def raw(self):
        return None
