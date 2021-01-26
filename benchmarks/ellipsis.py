from ndindex import ellipsis

class Timeellipsis:
    def setup(self):
        self.e = ellipsis()

    def time_constructor_int(self):
        ellipsis()

    def time_reduce(self):
        self.e.reduce()

    def time_reduce_shape(self):
        self.e.reduce(10)

    def time_newshape(self):
        self.e.newshape((10, 5))

    def time_isempty(self):
        self.e.isempty()

    def time_isempty_shape(self):
        self.e.isempty((10, 5))
