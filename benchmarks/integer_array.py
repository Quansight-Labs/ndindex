import numpy as np
from ndindex import IntegerArray

class TimeIntegerArray:
    def setup(self):
        self.ia = IntegerArray([[1, 2], [2, -1]]*100)

    def time_constructor_list(self):
        IntegerArray([[1, 2], [2, -1]]*100)

    def time_constructor_array(self):
        IntegerArray(self.ia.array)

    def time_constructor_invalid(self):
        try:
            IntegerArray(np.array([0.5]))
        except TypeError:
            pass

    def time_reduce(self):
        self.ia.reduce()

    def time_reduce_shape(self):
        self.ia.reduce(10)

    def time_reduce_shape_error(self):
        try:
            self.ia.reduce(2)
        except IndexError:
            pass

    def time_newshape(self):
        self.ia.newshape((10, 5))

    def time_isempty(self):
        self.ia.isempty()

    def time_isempty_shape(self):
        self.ia.isempty((10, 5))
