from operator import index

class NDIndex:
    """
    Represents an index into an nd-array (i.e., a numpy array)
    """
    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"

    def __hash__(self):
        return hash(self.args)

class Slice(NDIndex):
    """
    Represents a slice on an axis of an nd-array
    """
    def __new__(cls, start, stop=None, step=None):
        # Canonicalize
        if step is None:
            step = 1
        if step == 0:
            raise ValueError("slice step cannot be zero")
        if stop is None:
            start, stop = 0, start

        start = index(start)
        stop = index(stop)
        step = index(step)

        args = (start, stop, step)

        r = range(start, stop, step)
        if len(r) == 1:
            return Integer(r[0])
        return super().__new__(cls, *args)

    @property
    def raw(self):
        return slice(*self.args)

class Integer(NDIndex):
    """
    Represents an integer index on an axis of an nd-array
    """
    def __new__(cls, idx):
        idx = index(idx)

        return super().__new__(cls, idx)

    @property
    def raw (self):
        return self.args[0]
