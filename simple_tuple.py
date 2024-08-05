import sys

from ndindex import (ndindex, Integer, BooleanArray, Slice, Tuple, ellipsis,
                     Newaxis, broadcast_shapes, BroadcastError)
from ndindex.array import ArrayIndex

from simple_slice import ImmutableObject

def _is_boolean_scalar(idx):
    return isinstance(idx, BooleanArray) and idx.shape == ()

class SimpleTuple(ImmutableObject):
    __slots__ = ()

    def _typecheck(self, *args):
        if 'numpy' in sys.modules:
            from numpy import ndarray, bool_
        else:
            ndarray, bool_ = (), () # pragma: no cover
        newargs = []
        arrays = []
        array_block_start = False
        array_block_stop = False
        has_array = any(isinstance(i, (ArrayIndex, list, ndarray, bool, bool_)) for i in args)
        has_boolean_scalar = False
        for arg in args:
            newarg = ndindex(arg)
            if isinstance(newarg, Tuple):
                if len(args) == 1:
                    raise ValueError("tuples inside of tuple indices are not supported. Did you mean to call Tuple(*args) instead of Tuple(args)?")
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
                    # If the arrays in a tuple index are separated by a slice,
                    # ellipsis, or newaxis, the behavior is that the
                    # dimensions indexed by the array (and integer) indices
                    # are added to the front of the final array shape. Travis
                    # told me that he regrets implementing this behavior in
                    # NumPy and that he wishes it were in error. So for now,
                    # that is what we are going to do, unless it turns out
                    # that we actually need it.
                    raise NotImplementedError("Array indices separated by slices, ellipses (...), or newaxes (None) are not supported")

        if newargs.count(...) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if len(arrays) > 0:
            if has_boolean_scalar:
                raise NotImplementedError("Tuples mixing boolean scalars (True or False) with arrays are not yet supported.")

            try:
                broadcast_shapes(*[i.shape for i in arrays])
            except BroadcastError:
                # This matches the NumPy error message. The BroadcastError has
                # a better error message, but it will be shown in the chained
                # traceback.
                raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes %s" % ' '.join([str(i.shape) for i in arrays]))

        return tuple(newargs)
