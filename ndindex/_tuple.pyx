import sys

# Forward declarations
cdef extern from *:
    """
    #ifndef NDINDEX_H
    #define NDINDEX_H

    typedef struct {
        PyObject_HEAD
    } NDIndex;

    typedef struct {
        NDIndex base;
    } ArrayIndex;

    typedef struct {
        NDIndex base;
    } Integer;

    typedef struct {
        NDIndex base;
    } Slice;

    typedef struct {
        ArrayIndex base;
    } BooleanArray;

    typedef struct {
        ArrayIndex base;
    } IntegerArray;

    #endif
    """
    ctypedef struct NDIndex:
        pass
    ctypedef struct ArrayIndex:
        NDIndex base
    ctypedef struct Integer:
        NDIndex base
    ctypedef struct Slice:
        NDIndex base
    ctypedef struct BooleanArray:
        ArrayIndex base
    ctypedef struct IntegerArray:
        ArrayIndex base

cdef object _ndindex, _ArrayIndex, _Integer, _Slice, _BooleanArray, _IntegerArray, _ellipsis, _Newaxis
cdef object _broadcast_shapes, _BroadcastError

# We cannot just add these imports to the top because of circular import
# issues. We can put them inside the constructor, but then they create a big
# performance bottleneck.
cdef void _lazy_import():
    global _ndindex, _ArrayIndex, _Integer, _Slice, _BooleanArray, _IntegerArray, _ellipsis, _Newaxis
    global _broadcast_shapes, _BroadcastError

    if _ndindex is None:
        from ndindex import ndindex, Integer, Slice, BooleanArray, IntegerArray, ellipsis, Newaxis
        from ndindex.array import ArrayIndex
        from ndindex.shapetools import broadcast_shapes, BroadcastError
        _ndindex = ndindex
        _ArrayIndex = ArrayIndex
        _Integer = Integer
        _Slice = Slice
        _BooleanArray = BooleanArray
        _IntegerArray = IntegerArray
        _ellipsis = ellipsis
        _Newaxis = Newaxis
        _broadcast_shapes = broadcast_shapes
        _BroadcastError = BroadcastError

cdef int _is_boolean_scalar(object idx):
    cdef object BooleanArray
    _lazy_import()
    BooleanArray = _BooleanArray
    return isinstance(idx, BooleanArray) and idx.shape == ()

cdef class _Tuple:
    cdef readonly tuple args

    def __cinit__(self, *args):
        self._typecheck(args)

    cdef inline void _typecheck(self, tuple args) except *:
        cdef:
            list newargs = []
            list arrays = []
            int array_block_start = 0
            int array_block_stop = 0
            int has_array = 0
            int has_boolean_scalar = 0
            object arg, newarg

        _lazy_import()

        if 'numpy' in sys.modules:
            ndarray = sys.modules['numpy'].ndarray
            bool_ = sys.modules['numpy'].bool_
        else:
            ndarray = bool_ = () # pragma: no cover

        has_array = any(isinstance(i, (_ArrayIndex, list, ndarray, bool, bool_)) for i in args)

        n_ellipses = 0
        for arg in args:
            newarg = _ndindex(arg)
            if isinstance(newarg, _ellipsis):
                n_ellipses += 1
                if n_ellipses > 1:
                    raise IndexError("an index can only have a single ellipsis ('...')")
            if isinstance(newarg, _Tuple):
                if len(args) == 1:
                    raise ValueError("tuples inside of tuple indices are not supported. Did you mean to call Tuple(*args) instead of Tuple(args)?")
                raise ValueError("tuples inside of tuple indices are not supported. If you meant to use a fancy index, use a list or array instead.")
            newargs.append(newarg)
            if isinstance(newarg, _ArrayIndex):
                array_block_start = 1
                if _is_boolean_scalar(newarg):
                    has_boolean_scalar = 1
                elif isinstance(newarg, _BooleanArray):
                    arrays.extend(newarg.raw.nonzero())
                else:
                    arrays.append(newarg.raw)
            elif has_array and isinstance(newarg, _Integer):
                array_block_start = 1
            if isinstance(newarg, (_Slice, _ellipsis, _Newaxis)) and array_block_start:
                array_block_stop = 1
            elif isinstance(newarg, (_ArrayIndex, _Integer)):
                if array_block_start and array_block_stop:
                    # If the arrays in a tuple index are separated by a slice,
                    # ellipsis, or newaxis, the behavior is that the
                    # dimensions indexed by the array (and integer) indices
                    # are added to the front of the final array shape. Travis
                    # told me that he regrets implementing this behavior in
                    # NumPy and that he wishes it were in error. So for now,
                    # that is what we are going to do, unless it turns out
                    # that we actually need it.
                    raise NotImplementedError("Array indices separated by slices, ellipses (...), or newaxes (None) are not supported")

        if len(arrays) > 0:
            if has_boolean_scalar:
                raise NotImplementedError("Tuples mixing boolean scalars (True or False) with arrays are not yet supported.")

            try:
                _broadcast_shapes(*[i.shape for i in arrays])
            except _BroadcastError:
                # This matches the NumPy error message. The BroadcastError has
                # a better error message, but it will be shown in the chained
                # traceback.
                raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes %s" % ' '.join([str(i.shape) for i in arrays]))

        self.args = tuple(newargs)

    def __getnewargs__(self):
        return self.args

    def __setstate__(self, state):
        pass

    def __getstate__(self):
        return ()

    @property
    def raw(self):
        return tuple(arg.raw for arg in self.args)

    # def __repr__(self):
    #     return f"_Tuple{self.args}"

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.args == other
        elif isinstance(other, _Tuple):
            return self.args == other.args
        return False

    def __ne__(self, other):
        return not self == other
