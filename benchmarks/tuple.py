from ndindex import Tuple

class TimeTuple:
    def setup(self):
        self.t = Tuple(slice(0, 10), ..., 1)
        self.t_arrays = Tuple([[0, 1], [1, 2]], 2, [0, 1])
        self.t_boolean_scalars = Tuple(True, 0, False)

    def time_constructor_builtin(self):
        Tuple(slice(0, 10), ..., 1)

    def time_constructor_ndindex(self):
        Tuple(*self.t.args)

    def time_constructor_boolean_scalars(self):
        Tuple(True, 0, False)

    def time_constructor_arrays(self):
        Tuple([[0, 1], [1, 2]], 2, [0, 1])

    def time_constructor_invalid_ellipses(self):
        try:
            Tuple(0, ..., ...)
        except IndexError:
            pass

    def time_constructor_invalid_array_broadcast(self):
        try:
            Tuple([0, 1], [0, 1, 2])
        except IndexError:
            pass

    def time_args(self):
        self.t.args
        self.t_arrays.args
        self.t_boolean_scalars.args

    def time_raw(self):
        self.t.raw
        self.t_arrays.raw
        self.t_boolean_scalars.raw

    def time_reduce(self):
        self.t.reduce()

    def time_reduce_boolean_scalars(self):
        self.t_boolean_scalars.reduce()

    def time_reduce_shape(self):
        self.t.reduce((10, 4, 2))

    def time_reduce_shape_error(self):
        try:
            self.t.reduce(10)
        except IndexError:
            pass

    def time_newshape(self):
        self.t.newshape((10, 4, 2))

    def time_expand(self):
        self.t.expand((10, 4, 2))

    def time_expand_arrays(self):
        self.t_arrays.expand((10, 4, 2))

    def time_expand_boolean_scalars(self):
        self.t_boolean_scalars.expand((10, 4, 2))

    def time_broadcast_arrays_no_arrays(self):
        self.t.broadcast_arrays()

    def time_broadcast_arrays(self):
        self.t_arrays.broadcast_arrays()

    def time_isempty(self):
        self.t.isempty()

    def time_isempty_shape(self):
        self.t.isempty((10, 4, 2))
