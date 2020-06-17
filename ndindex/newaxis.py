from .ndindex import NDIndex


class Newaxis(NDIndex):
    """
    Increases the dimension of the existing array by one more dimension.

    >>> from ndindex import Newaxis
    >>> import numpy as np
    >>> a = np.arange(0,6).reshape(2,3)
    >>> a[Newaxis().raw].shape
    (1, 2, 3)
    >>> a[:, Newaxis().raw, :].shape
    (2, 1, 3)

    Using `Newaxis().raw` as an index is equivalent to using `numpy.newaxis`.
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
