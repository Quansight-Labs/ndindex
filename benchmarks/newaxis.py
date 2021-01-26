from ndindex import Newaxis

class TimeNewaxis:
    def setup(self):
        self.n = Newaxis()

    def time_constructor_int(self):
        Newaxis()

    def time_reduce(self):
        self.n.reduce()

    def time_reduce_shape(self):
        self.n.reduce(10)

    def time_newshape(self):
        self.n.newshape((10, 5))

    def time_isempty(self):
        self.n.isempty()

    def time_isempty_shape(self):
        self.n.isempty((10, 5))
