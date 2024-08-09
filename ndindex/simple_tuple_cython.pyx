# cython: language_level=3
# distutils: language = c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from cpython.ref cimport PyObject
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

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

cdef void _lazy_import() nogil:
    global _ndindex, _ArrayIndex, _Integer, _Slice, _BooleanArray, _IntegerArray, _ellipsis, _Newaxis
    global _broadcast_shapes, _BroadcastError

    with gil:
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

cdef class SimpleTupleCython:
    cdef readonly tuple args
    cdef PyObject** c_args
    cdef Py_ssize_t n_args

    def __cinit__(self, *args):
        self._typecheck(args)

    def __dealloc__(self):
        if self.c_args is not NULL:
            free(self.c_args)

    cdef inline void _typecheck(self, tuple args) except *:
        cdef:
            list newargs = []
            list arrays = []
            int array_block_start = 0
            int array_block_stop = 0
            int has_array = 0
            int has_boolean_scalar = 0
            Py_ssize_t i
            object arg, newarg

        _lazy_import()

        self.n_args = len(args)

        # Check for numpy availability
        if 'numpy' in sys.modules:
            ndarray = sys.modules['numpy'].ndarray
            bool_ = sys.modules['numpy'].bool_
        else:
            ndarray = bool_ = ()

        # Check if any argument is an array-like object
        has_array = any(isinstance(i, (_ArrayIndex, list, ndarray, bool, bool_)) for i in args)

        for i in range(self.n_args):
            arg = args[i]
            newarg = _ndindex(arg)
            if isinstance(newarg, SimpleTupleCython):
                if self.n_args == 1:
                    raise ValueError("tuples inside of tuple indices are not supported. Did you mean to call SimpleTupleCython(*args) instead of SimpleTupleCython(args)?")
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
                    raise NotImplementedError("Array indices separated by slices, ellipses (...), or newaxes (None) are not supported")

        if newargs.count(_ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if len(arrays) > 0:
            if has_boolean_scalar:
                raise NotImplementedError("Tuples mixing boolean scalars (True or False) with arrays are not yet supported.")

            try:
                _broadcast_shapes(*[i.shape for i in arrays])
            except _BroadcastError:
                raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes %s" % ' '.join([str(i.shape) for i in arrays]))

        self.args = tuple(newargs)
        self.c_args = <PyObject**>malloc(self.n_args * sizeof(PyObject*))
        if self.c_args is NULL:
            raise MemoryError()
        for i in range(self.n_args):
            self.c_args[i] = <PyObject*>newargs[i]

    @property
    def raw(self):
        return tuple((<object>self.c_args[i]).raw for i in range(self.n_args))

    def __repr__(self):
        return f"SimpleTupleCython{self.args}"

    def __eq__(self, other):
        if not isinstance(other, SimpleTupleCython):
            return False
        return self.args == other.args

    def __ne__(self, other):
        return not self == other
