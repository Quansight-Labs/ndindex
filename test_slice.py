from ndindex import Slice
from simple_slice import SimpleSlice, SimpleSliceSubclass, SimpleSliceCythonSubclass, SimpleSlicePybind11Subclass, SimpleSliceRustSubclass
from simple_slice_cython import SimpleSliceCython
from simple_slice_pybind11 import SimpleSlicePybind11
from simple_slice_rust import SimpleSliceRust

from numpy import bool_

from pytest import raises, mark

@mark.parametrize('SliceClass',
                  [Slice, SimpleSlice, SimpleSliceCython, SimpleSlicePybind11,
                   SimpleSliceRust, SimpleSliceSubclass,
                   SimpleSliceCythonSubclass,
                     SimpleSlicePybind11Subclass, SimpleSliceRustSubclass

])
def test_slice_args(SliceClass):
    # Test the behavior when not all three arguments are given
    # TODO: Incorporate this into the normal slice tests
    raises(TypeError, lambda: slice())
    raises(TypeError, lambda: SliceClass())
    raises(TypeError, lambda: SliceClass(1.0))
    raises(TypeError, lambda: SliceClass('1'))
    # See docstring of operator_index()
    raises(TypeError, lambda: SliceClass(True))
    raises(TypeError, lambda: SliceClass(bool_(True)))

    S = SliceClass(1)
    assert S == SliceClass(S) == SliceClass(None, 1) == SliceClass(None, 1, None) == SliceClass(None, 1, None)
    # assert S.raw == slice(None, 1, None)
    assert S.args == (S.start, S.stop, S.step)

    S = SliceClass(0, 1)
    assert S == SliceClass(S) == SliceClass(0, 1, None)
    # assert S.raw == slice(0, 1, None)
    assert S.args == (S.start, S.stop, S.step)

    S = SliceClass(0, 1, 2)
    assert S == SliceClass(S)
    # assert S.raw == slice(0, 1, 2)
    assert S.args == (S.start, S.stop, S.step)
