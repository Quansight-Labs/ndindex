from ndindex import Slice

class TimeSlice:
    def setup(self):
        self.s1 = Slice(0, 30)
        self.s2 = Slice(0, 1, 10)

    def time_constructor_slice(self):
        Slice(slice(0, 30))

    def time_constructor_ints(self):
        Slice(0, 1, 10)

    def time_constructor_invalid(self):
        try:
            Slice(0.5)
        except TypeError:
            pass

    def time_reduce(self):
        self.s1.reduce()
        self.s2.reduce()

    def time_reduce_shape(self):
        self.s1.reduce(10)
        self.s2.reduce(10)

    def time_newshape(self):
        self.s1.newshape((10, 5))
        self.s2.newshape((10, 5))

    def time_isempty(self):
        self.s1.isempty()

    def time_isempty_shape(self):
        self.s1.isempty((10, 5))

    def time_len(self):
        len(self.s1)
        len(self.s2)
