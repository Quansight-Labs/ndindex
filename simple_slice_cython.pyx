# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from cpython cimport PyIndex_Check
from libc.stdint cimport int64_t
import sys

cdef extern from "Python.h":
    int PyIndex_Check(object obj)
    object PyNumber_Index(object obj)

cdef inline int64_t operator_index(object idx) except? -1:
    if PyIndex_Check(idx):
        return PyNumber_Index(idx)
    if isinstance(idx, bool):
        raise TypeError("'bool' object cannot be interpreted as an integer")
    if 'numpy' in sys.modules and isinstance(idx, sys.modules['numpy'].bool_):
        raise TypeError("'np.bool_' object cannot be interpreted as an integer")
    return PyNumber_Index(idx)

cdef class default:
    pass

cdef class SimpleSlice:
    cdef readonly int64_t start
    cdef readonly int64_t stop
    cdef readonly int64_t step
    cdef readonly bint has_start
    cdef readonly bint has_stop
    cdef readonly bint has_step

    def __cinit__(self, object start, object stop=default, object step=default):
        self._typecheck(start, stop, step)

    cdef inline void _typecheck(self, object start, object stop, object step) except *:
        cdef int64_t _start, _stop, _step

        if isinstance(start, SimpleSlice):
            self.start = (<SimpleSlice>start).start
            self.stop = (<SimpleSlice>start).stop
            self.step = (<SimpleSlice>start).step
            self.has_start = (<SimpleSlice>start).has_start
            self.has_stop = (<SimpleSlice>start).has_stop
            self.has_step = (<SimpleSlice>start).has_step
            return

        if isinstance(start, slice):
            self._typecheck(start.start, start.stop, start.step)
            return

        if stop is default:
            start, stop = None, start

        self.has_start = start is not None
        self.has_stop = stop is not None
        self.has_step = step is not default

        if self.has_start:
            self.start = operator_index(start)
        if self.has_stop:
            self.stop = operator_index(stop)
        if self.has_step:
            self.step = operator_index(step)
            if self.step == 0:
                raise ValueError("slice step cannot be zero")

    def __repr__(self):
        cdef list parts = []
        if self.has_start:
            parts.append(f"start={self.start}")
        if self.has_stop:
            parts.append(f"stop={self.stop}")
        if self.has_step:
            parts.append(f"step={self.step}")
        return f"SimpleSlice({', '.join(parts)})"
