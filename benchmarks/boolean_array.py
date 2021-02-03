import numpy as np
from ndindex import BooleanArray

class TimeBooleanArray:
    def setup(self):
        self.ba = BooleanArray([[False, True], [True, False]]*100)

    def time_constructor_list(self):
        BooleanArray([[False, True], [True, False]]*100)

    def time_constructor_array(self):
        BooleanArray(self.ba.array)

    def time_constructor_bool(self):
        BooleanArray(True)

    def time_constructor_invalid(self):
        try:
            BooleanArray(np.array([0]))
        except TypeError:
            pass

    def time_reduce(self):
        self.ba.reduce()

    def time_reduce_shape(self):
        self.ba.reduce((200, 2, 5))

    def time_reduce_shape_error(self):
        try:
            self.ba.reduce(10)
        except IndexError:
            pass

    def time_newshape(self):
        self.ba.newshape((200, 2, 5))

    def time_isempty(self):
        self.ba.isempty()

    def time_isempty_shape(self):
        self.ba.isempty((200, 2, 5))
