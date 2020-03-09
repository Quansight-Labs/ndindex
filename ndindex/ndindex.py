from operator import index

class NDIndex:
    """
    Represents an index into an nd-array (i.e., a numpy array)
    """
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"

    def __hash__(self):
        return hash(self.args)

class Slice(NDIndex):
    """
    Represents a slice on an axis of an nd-array
    """
    def __init__(self, start, stop=None, step=None):
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

        self.args = (start, stop, step)

    @property
    def raw(self):
        return slice(*self.args)
