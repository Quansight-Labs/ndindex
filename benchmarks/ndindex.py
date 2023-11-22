"""
Benchmarks for the ndindex() function
"""

from ndindex import ndindex
import numpy as np

class NDIndexTypes:
    def setup(self):
        from ndindex import (Slice, Tuple, Integer, ellipsis, Newaxis,
                             IntegerArray, BooleanArray)
        self.slice = Slice(0, 4, 2)
        self.integer = Integer(1)
        self.tuple = Tuple(self.slice, ..., 0)
        self.ellipsis = ellipsis()
        self.newaxis = Newaxis()
        self.integer_array = IntegerArray([[1, 2], [-1, 2]])
        self.boolean_array = BooleanArray([[True, False], [False, False]])

    def time_ndindex_Slice(self):
        ndindex(self.slice)

    def time_ndindex_Integer(self):
        ndindex(self.integer)

    def time_ndindex_Tuple(self):
        ndindex(self.tuple)

    def time_ndindex_ellipsis(self):
        ndindex(self.ellipsis)

    def time_ndindex_Newaxis(self):
        ndindex(self.newaxis)

    def time_ndindex_IntegerArray(self):
        ndindex(self.integer_array)

    def time_ndindex_BooleanArray(self):
        ndindex(self.boolean_array)

class BuiltinTypes:
    def setup(self):
        self.int64 = np.int64(1)
        self.integer_array = np.array([[1, 2], [-1, 2]])
        self.boolean_array = np.array([[True, False], [False, False]])
        self.bool_ = np.bool_(False)
        self.tuple = (slice(0, 4, 2), ..., 1)

    def time_ndindex_slice(self):
        ndindex[0:4:2]

    def time_ndindex_int(self):
        ndindex(1)

    def time_ndindex_int64(self):
        ndindex(self.int64)

    def time_ndindex_tuple(self):
        ndindex(self.tuple)

    def time_ndindex_Ellipsis(self):
        ndindex(...)

    def time_ndindex_newaxis(self):
        ndindex(None)

    def time_ndindex_integer_array(self):
        ndindex(self.integer_array)

    def time_ndindex_boolean_array(self):
        ndindex(self.boolean_array)

    def time_ndindex_bool(self):
        ndindex(False)

    def time_ndindex_bool_(self):
        ndindex(self.bool_)
