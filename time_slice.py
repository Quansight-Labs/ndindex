import timeit
import random
from ndindex import Slice
from simple_slice import (SimpleSlice, SimpleSliceSubclass, SimpleSliceCythonSubclass,
                          SimpleSliceRustSubclass,
                          SimpleSlicePybind11Subclass)

from simple_slice_cython import SimpleSliceCython
from simple_slice_pybind11 import SimpleSlicePybind11
from simple_slice_rust import SimpleSliceRust

from IPython.core.magics.execution import _format_time


slice_classes = [Slice,
                 SimpleSlice, SimpleSliceSubclass,
                 SimpleSliceCython, SimpleSliceCythonSubclass,
                 SimpleSlicePybind11, SimpleSlicePybind11Subclass,
                 SimpleSliceRust, SimpleSliceRustSubclass,
                 ]
slice_classes_without_reduce = [SimpleSliceCython, SimpleSlicePybind11,
                                SimpleSliceRust]

N_RUNS = 1_000_000

def random_value():
    return random.choice([None] + list(range(-100, -1)) + list(range(1, 100)))

def generate_inputs(n=N_RUNS):
    return [(random_value(), random_value(), random_value()) for _ in range(n)]

def benchmark_creation(SliceClass, inputs):
    for start, stop, step in inputs:
        SliceClass(start, stop, step)

def benchmark_args(SliceClass, n=N_RUNS):
    s = SliceClass(0, 100, 2)
    for _ in range(n):
        start, stop, step = s.args

def bench_reduce(SliceClass, inputs):
    for start, stop, step in inputs:
        s = SliceClass(start, stop, step)
        s.reduce()

def bench_reduce_shape(SliceClass, inputs):
    for start, stop, step in inputs:
        s = SliceClass(start, stop, step)
        s.reduce((150,))

def run_benchmark(name, func, SliceClass, *args):
    time = timeit.timeit(lambda: func(SliceClass, *args), number=1)
    print(f"{SliceClass.__name__:<30}: {_format_time(time/N_RUNS)}")

def main():
    print("Benchmarking SimpleSlice implementations")
    print("----------------------------------------")

    inputs = generate_inputs()

    print("\nObject Creation:")
    for SliceClass in slice_classes:
        run_benchmark("Creation", benchmark_creation, SliceClass, inputs)

    print("\nArgs Access:")
    for SliceClass in slice_classes:
        run_benchmark("Access", benchmark_args, SliceClass)

    print("\nReduce:")
    for SliceClass in slice_classes:
        if SliceClass in slice_classes_without_reduce:
            continue
        run_benchmark("Reduce", bench_reduce, SliceClass, inputs)

    print("\nReduce with shape:")
    for SliceClass in slice_classes:
        if SliceClass in slice_classes_without_reduce:
            continue
        run_benchmark("Reduce Shape", bench_reduce_shape, SliceClass, inputs)

if __name__ == "__main__":
    main()
