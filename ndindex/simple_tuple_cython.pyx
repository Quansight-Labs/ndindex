import sys

cdef class SimpleTupleCython:
    cdef readonly tuple args

    def __cinit__(self, *args):
        self._typecheck(args)

    cdef inline void _typecheck(self, tuple args) except *:
        cdef list newargs = []
        cdef list arrays = []
        cdef bint array_block_start = False
        cdef bint array_block_stop = False
        cdef bint has_array = False
        cdef bint has_boolean_scalar = False


        from .ndindex import ndindex
        from .array import ArrayIndex
        from .ellipsis import ellipsis
        from .newaxis import Newaxis
        from .slice import Slice
        from .integer import Integer
        from .booleanarray import BooleanArray
        from .integerarray import IntegerArray

        from .shapetools import broadcast_shapes, BroadcastError

        # Check for numpy availability
        if 'numpy' in sys.modules:
            ndarray = sys.modules['numpy'].ndarray
            bool_ = sys.modules['numpy'].bool_
        else:
            ndarray = bool_ = ()

        # Check if any argument is an array-like object
        has_array = any(isinstance(i, (ArrayIndex, list, ndarray, bool, bool_)) for i in args)

        for arg in args:
            newarg = ndindex(arg)
            if isinstance(newarg, SimpleTupleCython):
                if len(args) == 1:
                    raise ValueError("tuples inside of tuple indices are not supported. Did you mean to call SimpleTupleCython(*args) instead of SimpleTupleCython(args)?")
                raise ValueError("tuples inside of tuple indices are not supported. If you meant to use a fancy index, use a list or array instead.")
            newargs.append(newarg)
            if isinstance(newarg, ArrayIndex):
                array_block_start = True
                if _is_boolean_scalar(newarg):
                    has_boolean_scalar = True
                elif isinstance(newarg, BooleanArray):
                    arrays.extend(newarg.raw.nonzero())
                else:
                    arrays.append(newarg.raw)
            elif has_array and isinstance(newarg, Integer):
                array_block_start = True
            if isinstance(newarg, (Slice, ellipsis, Newaxis)) and array_block_start:
                array_block_stop = True
            elif isinstance(newarg, (ArrayIndex, Integer)):
                if array_block_start and array_block_stop:
                    raise NotImplementedError("Array indices separated by slices, ellipses (...), or newaxes (None) are not supported")

        if newargs.count(...) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if len(arrays) > 0:
            if has_boolean_scalar:
                raise NotImplementedError("Tuples mixing boolean scalars (True or False) with arrays are not yet supported.")

            try:
                broadcast_shapes(*[i.shape for i in arrays])
            except BroadcastError:
                raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes %s" % ' '.join([str(i.shape) for i in arrays]))

        self.args = tuple(newargs)

    @property
    def raw(self):
        return tuple(i.raw for i in self.args)

    def __repr__(self):
        return f"SimpleTupleCython{self.args}"

    def __eq__(self, other):
        if not isinstance(other, SimpleTupleCython):
            return False
        return self.args == other.args

    def __ne__(self, other):
        return not self == other

cdef bint _is_boolean_scalar(object idx):
    from .booleanarray import BooleanArray
    return isinstance(idx, BooleanArray) and idx.shape == ()
