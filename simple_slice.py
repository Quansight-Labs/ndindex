import numbers
from collections.abc import Sequence
import operator
import sys

from simple_slice_cython import SimpleSliceCython
from simple_slice_pybind11 import SimpleSlicePybind11
from simple_slice_rust import SimpleSliceRust

def asshape(shape, axis=None, *, allow_int=True, allow_negative=False):
    # from .integer import Integer
    # from .tuple import Tuple
    # if isinstance(shape, (Tuple, Integer)):
    #     raise TypeError("ndindex types are not meant to be used as a shape - "
    #                     "did you mean to use the built-in tuple type?")

    if isinstance(shape, numbers.Number):
        if allow_int:
            shape = (operator_index(shape),)
        else:
            raise TypeError(f"expected sequence of integers, not {type(shape).__name__}")

    if not isinstance(shape, Sequence) or isinstance(shape, str):
        raise TypeError("expected sequence of integers" + allow_int*" or a single integer" + ", not " + type(shape).__name__)
    l = len(shape)

    newshape = []
    # numpy uses __getitem__ rather than __iter__ to index into shape, so we
    # match that
    for i in range(l):
        # Raise TypeError if invalid
        val = shape[i]
        if val is None:
            raise ValueError("unknonwn (None) dimensions are not supported")

        newshape.append(operator_index(shape[i]))

        if not allow_negative and val < 0:
            raise ValueError("unknown (negative) dimensions are not supported")

    if axis is not None:
        if len(newshape) <= axis:
            raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional, but {axis + 1} were indexed")

    return tuple(newshape)

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

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(repr, self.args))})"

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        if not isinstance(other, ImmutableObject):
            try:
                other = self.__class__(other)
            except TypeError:
                return False

        return self.args == other.args

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

class SimpleSliceSubclass(SimpleSlice):
    pass

class SimpleSliceCythonSubclass(SimpleSliceCython):
    pass

class SimpleSlicePybind11Subclass(SimpleSlicePybind11):
    pass

class SimpleSliceRustSubclass(SimpleSliceRust):
    pass

def reduce(self, shape=None, *, axis=0, negative_int=False):
    start, stop, step = self.args

    # Canonicalize with no shape

    if step is None:
        step = 1
    if start is None:
        if step > 0:
            start = 0
        else: # step < 0
            start = -1

    if start is not None and stop is not None:
        if start >= 0 and stop >= 0 or start < 0 and stop < 0:
            if step > 0:
                if stop <= start:
                    start, stop, step = 0, 0, 1
                elif start >= 0 and start + step >= stop:
                    # Indexes 1 element. Start has to be >= 0 because a
                    # negative start could be less than the size of the
                    # axis, in which case it will clip and the single
                    # element will be element 0. We can only do that
                    # reduction if we know the shape.

                    # Note that returning Integer here is wrong, because
                    # slices keep the axis and integers remove it.
                    stop, step = start + 1, 1
                elif start < 0 and start + step > stop:
                    # The exception is this case where stop is already
                    # start + 1.
                    step = stop - start
                if start >= 0:
                    stop -= (stop - start - 1) % step
            else: # step < 0
                if stop >= start:
                    start, stop, step = 0, 0, 1
                elif start < 0 and start + step <= stop:
                    if start < -1:
                        stop, step = start + 1, 1
                    else: # start == -1
                        stop, step = start - 1, -1
                elif stop == start - 1:
                    stop, step = start + 1, 1
                elif start >= 0 and start + step <= stop:
                    # Indexes 0 or 1 elements. We can't change stop
                    # because start might clip to a smaller true start if
                    # the axis is smaller than it, and increasing stop
                    # would prevent it from indexing an element in that
                    # case. The exception is the case right before this
                    # one (stop == start - 1). In that case start cannot
                    # clip past the stop (it always indexes the same one
                    # element in the cases where it indexes anything at
                    # all).
                    step = stop - start
                if start < 0:
                    stop -= (stop - start + 1) % step
        elif start >= 0 and stop < 0 and step < 0 and (start < -step or
                                                       -stop - 1 < -step):
            if stop == -1:
                start, stop, step = 0, 0, 1
            else:
                step = max(-start - 1, stop + 1)
        elif start < 0 and stop == 0 and step > 0:
            start, stop, step = 0, 0, 1
        elif start < 0 and stop >= 0 and step >= min(-start, stop):
            step = min(-start, stop)
            if start == -1 or stop == 1:
                # Can only index 0 or 1 elements. We can either pick a
                # version with positive start and negative step, or
                # negative start and positive step. We prefer the former
                # as it matches what is done for reduce() with a shape
                # (start is always nonnegative).
                assert step == 1
                start, stop, step = stop - 1, start - 1, -1
    elif start is not None and stop is None:
        if start == -1 and step > 0:
            start, stop, step = (-1, -2, -1)
        elif start < 0 and step >= -start:
            step = -start
        elif step < 0:
            if start == 0:
                start, stop, step = 0, 1, 1
            elif 0 <= start < -step:
                step = -start - 1
    if shape is None:
        return type(self)(start, stop, step)

    # Further canonicalize with an explicit array shape

    shape = asshape(shape, axis=axis)
    size = shape[axis]

    if stop is None:
        if step > 0:
            stop = size
        else:
            stop = -size - 1

    if stop < -size:
        stop = -size - 1

    if size == 0:
        start, stop, step = 0, 0, 1
    elif step > 0:
        # start cannot be None
        if start < 0:
            start = size + start
        if start < 0:
            start = 0
        if start >= size:
            start, stop, step = 0, 0, 1

        if stop < 0:
            stop = size + stop
            if stop < 0:
                stop = 0
        else:
            stop = min(stop, size)
        stop -= (stop - start - 1) % step

        if stop - start == 1:
            # Indexes 1 element.
            step = 1
        elif stop - start <= 0:
            start, stop, step = 0, 0, 1
    else:
        if start < 0:
            if start >= -size:
                start = size + start
            else:
                start, stop = 0, 0
        if start >= 0:
            start = min(size - 1, start)

        if -size <= stop < 0:
            stop += size

        if stop >= 0:
            if start - stop == 1:
                stop, step = start + 1, 1
            elif start - stop <= 0:
                start, stop, step = 0, 0, 1
            else:
                stop += (start - stop - 1) % -step

        # start >= 0
        if (stop < 0 and start - size - stop <= -step
            or stop >= 0 and start - stop <= -step):
            stop, step = start + 1, 1
        if stop < 0 and start % step != 0:
            # At this point, negative stop is only necessary to index the
            # first element. If that element isn't actually indexed, we
            # prefer a nonnegative stop. Otherwise, stop will be -size - 1.
            stop = start % -step - 1
    return self.__class__(start, stop, step)

SimpleSlice.reduce = reduce
SimpleSliceSubclass.reduce = reduce
SimpleSliceCythonSubclass.reduce = reduce
SimpleSlicePybind11Subclass.reduce = reduce
SimpleSliceRustSubclass.reduce = reduce
