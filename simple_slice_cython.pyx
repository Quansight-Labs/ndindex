# distutils: language = c
# cython: language_level=3

import operator
import sys
from cpython cimport PyObject

cdef extern from "Python.h":
    int PyIndex_Check(object obj)

cdef object operator_index(object idx):
    if isinstance(idx, bool):
        raise TypeError("'bool' object cannot be interpreted as an integer")
    if 'numpy' in sys.modules and isinstance(idx, sys.modules['numpy'].bool_):
        raise TypeError("'np.bool_' object cannot be interpreted as an integer")
    return operator.index(idx)

cdef class default:
    pass

cdef class SimpleSlice:
    cdef readonly tuple args

    def __cinit__(self, start, stop=default, step=None):
        self.args = self._typecheck(start, stop, step)

    cpdef tuple _typecheck(self, start, stop=default, step=None):
        if isinstance(start, SimpleSlice):
            return start.args
        if isinstance(start, slice):
            start, stop, step = start.start, start.stop, start.step
        if stop is default:
            start, stop = None, start
        if step == 0:
            raise ValueError("slice step cannot be zero")
        if start is not None:
            start = operator_index(start)
        if stop is not None:
            stop = operator_index(stop)
        if step is not None:
            step = operator_index(step)
        return (start, stop, step)

    @property
    def start(self):
        return self.args[0]

    @property
    def stop(self):
        return self.args[1]

    @property
    def step(self):
        return self.args[2]

    def __repr__(self):
        return f"SimpleSlice{self.args}"
