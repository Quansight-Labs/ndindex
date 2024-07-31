# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from cpython cimport PyObject
from libc.stdint cimport int64_t
import sys

cdef extern from "Python.h":
    Py_ssize_t PyNumber_AsSsize_t(object obj, object exc) except? -1
    object PyNumber_Index(object obj)
    bint PyBool_Check(object obj)
    int64_t PyLong_AsLongLong(object obj) except? -1

cdef bint _NUMPY_IMPORTED = False
cdef type _NUMPY_BOOL = None

cdef inline bint is_numpy_bool(object obj):
    global _NUMPY_IMPORTED, _NUMPY_BOOL
    if not _NUMPY_IMPORTED:
        if 'numpy' in sys.modules:
            _NUMPY_BOOL = sys.modules['numpy'].bool_
        _NUMPY_IMPORTED = True
    return _NUMPY_BOOL is not None and isinstance(obj, _NUMPY_BOOL)

cdef inline int64_t cy_operator_index(object idx) except? -1:
    cdef object result

    if PyBool_Check(idx) or is_numpy_bool(idx):
        raise TypeError(f"'{type(idx).__name__}' object cannot be interpreted as an integer")

    return PyNumber_AsSsize_t(idx, IndexError)

cdef class default:
    pass

cdef class SimpleSliceCython:
    cdef readonly tuple args
    cdef int64_t _start
    cdef int64_t _stop
    cdef int64_t _step
    cdef bint _has_start
    cdef bint _has_stop
    cdef bint _has_step
    cdef bint _reduced

    def __cinit__(self, start, stop=default, step=None, _reduced=False):
        self._typecheck(start, stop, step, _reduced)

    cdef inline void _typecheck(self, object start, object stop, object step, object _reduced) except *:
        cdef object _start, _stop, _step

        if isinstance(start, SimpleSliceCython):
            self.args = (<SimpleSliceCython>start).args
            self._start = (<SimpleSliceCython>start)._start
            self._stop = (<SimpleSliceCython>start)._stop
            self._step = (<SimpleSliceCython>start)._step
            self._has_start = (<SimpleSliceCython>start)._has_start
            self._has_stop = (<SimpleSliceCython>start)._has_stop
            self._has_step = (<SimpleSliceCython>start)._has_step
            self._reduced = _reduced
            return

        if isinstance(start, slice):
            self._typecheck(start.start, start.stop, start.step, _reduced)
            return

        if stop is default:
            start, stop = None, start

        self._has_start = start is not None
        self._has_stop = stop is not None
        self._has_step = step is not None

        if self._has_start:
            self._start = cy_operator_index(start)
            _start = self._start
        else:
            _start = None

        if self._has_stop:
            self._stop = cy_operator_index(stop)
            _stop = self._stop
        else:
            _stop = None

        if self._has_step:
            self._step = cy_operator_index(step)
            if self._step == 0:
                raise ValueError("slice step cannot be zero")
            _step = self._step
        else:
            _step = None

        self.args = (_start, _stop, _step)

        self._reduced = _reduced

    @property
    def raw(self):
        return slice(self.start, self.stop, self.step)

    @property
    def start(self):
        return self.args[0]

    @property
    def stop(self):
        return self.args[1]

    @property
    def step(self):
        return self.args[2]

    @property
    def _reduced(self):
        return self._reduced

    def __repr__(self):
        return f"SimpleSliceCython{self.args}"

    def __eq__(self, other):
        if not isinstance(other, SimpleSliceCython):
            return False
        return self.args == other.args

    def __ne__(self, other):
        return not self == other
