import warnings

from numpy import ndarray, asarray, integer, bool_, array2string, empty, intp

from .ndindex import NDIndex, asshape

class ArrayIndex(NDIndex):
    """
    Super class for array indices

    This class should not be instantiated directly. Rather, use one of its
    subclasses, `IntegerArray` or `BooleanArray`.
    """
    # Subclasses should redefine this
    dtype = None

    def _typecheck(self, idx, shape=None):
        if self.dtype is None:
            raise TypeError("Do not instantiate the superclass ArrayIndex directly")

        if shape is not None:
            if idx != []:
                raise ValueError("The shape argument is only allowed for empty arrays (idx=[])")
            shape = asshape(shape)
            if 0 not in shape:
                raise ValueError("The shape argument must be an empty shape")
            idx = empty(shape, dtype=self.dtype)

        if isinstance(idx, (list, ndarray, bool, integer, int, bool_)):
            # Ignore deprecation warnings for things like [1, []]. These will be
            # filtered out anyway since they produce object arrays.
            with warnings.catch_warnings(record=True):
                a = asarray(idx)
                if a is idx:
                    a = a.copy()
                if isinstance(idx, list) and 0 in a.shape:
                    a = a.astype(self.dtype)
            if self.dtype == intp and issubclass(a.dtype.type, integer):
                if a.dtype != self.dtype:
                    a = a.astype(self.dtype)
            if a.dtype != self.dtype:
                raise TypeError(f"The input array to {self.__class__.__name__} must have dtype {self.dtype.__name__}, not {a.dtype}")
            a.flags.writeable = False
            return (a,)
        raise TypeError(f"{self.__class__.__name__} must be created with an array with dtype {self.dtype.__name__}")

    @property
    def raw(self):
        return self.args[0]

    @property
    def array(self):
        """
        Return the NumPy array of self.

        This is the same as `self.args[0]`.
        """
        return self.args[0]

    @property
    def shape(self):
        """
        Return the shape of the array of self.

        This is the same as self.array.shape. Note that this is **not** the
        same as the shape of an array that is indexed by self. Use
        :meth:`newshape` to get that.

        """
        return self.array.shape

    @property
    def ndim(self):
        """
        Return the number of dimensions of the array of self.

        This is the same as self.array.ndim. Note that this is **not** the
        same as the number of dimensions of an array that is indexed by self.
        Use `len` on :meth:`newshape` to get that.

        """
        return self.array.ndim

    # The repr form recreates the object. The str form gives the truncated
    # array string and is explicitly non-valid Python (doesn't have commas).
    def __repr__(self):
        if 0 not in self.shape:
            arg = repr(self.array.tolist())
        else:
            arg = f"[], shape={self.shape}"
        return f"{self.__class__.__name__}({arg})"

    def __str__(self):
        return (self.__class__.__name__
                + "("
                + array2string(self.array).replace('\n', '')
                + ")")

    def __hash__(self):
        return hash(self.array.tobytes())
