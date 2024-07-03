import operator
import sys

def operator_index(idx):
    if isinstance(idx, bool):
        raise TypeError("'bool' object cannot be interpreted as an integer")
    if 'numpy' in sys.modules and isinstance(idx, sys.modules['numpy'].bool_):
        raise TypeError("'np.bool_' object cannot be interpreted as an integer")
    return operator.index(idx)

class default:
    pass

class ImmutableObject:
    __slots__ = ('args',)

    def __init__(self, *args, **kwargs):
        args = self._typecheck(*args, **kwargs)
        self.args = args

class SimpleSlice(ImmutableObject):
    __slots__ = ()

    def _typecheck(self, start, stop=default, step=None):
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

        args = (start, stop, step)

        return args

    @property
    def raw(self):
        return slice(*self.args)

    @property
    def start(self):
        return self.args[0]

    @property
    def stop(self):
        return self.args[1]

    @property
    def step(self):
        return self.args[2]
