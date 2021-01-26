import numpy as np
from ndindex import Integer

class TimeInteger:
    def setup(self):
        self.i10 = Integer(10)
        self.in10 = Integer(-10)

    def time_constructor_int(self):
        Integer(0)

    def time_constructor_int64(self):
        Integer(np.int64(0))

    def time_constructor_invalid(self):
        try:
            Integer(0.5)
        except TypeError:
            pass

    def time_reduce(self):
        self.i10.reduce()

    def time_reduce_shape(self):
        self.in10.reduce(10)

    def time_reduce_shape_error(self):
        try:
            self.i10.reduce(10)
        except IndexError:
            pass

    def time_newshape(self):
        self.in10.newshape((10, 5))

    def time_isempty(self):
        self.in10.isempty()

    def time_isempty_shape(self):
        self.in10.isempty((10, 5))
