import timeit
import random
from ndindex import Slice
from simple_slice import SimpleSlice as PySimpleSlice
from simple_slice_cython import SimpleSlice as CySimpleSlice

N_RUNS = 1_000_000

def random_value():
    return random.choice([None] + list(range(-100, -1)) + list(range(1, 100)))

def benchmark_creation(SliceClass):
    SliceClass(random_value(), random_value(), random_value())

def benchmark_args(SliceClass,):
    s = SliceClass(0, 100, 2)
    start, stop, step = s.args

def run_benchmark(name, func, SliceClass):
    time = timeit.timeit(lambda: func(SliceClass), number=N_RUNS)
    print(f"{name} - {SliceClass.__module__}: {time/N_RUNS} seconds")

def main():
    print("Benchmarking SimpleSlice implementations")
    print("----------------------------------------")

    benchmarks = [
        ("Object Creation", benchmark_creation),
        ("Args Access", benchmark_args),
    ]

    for name, func in benchmarks:
        print(f"\n{name}:")
        run_benchmark(name, func, Slice)
        run_benchmark(name, func, PySimpleSlice)
        run_benchmark(name, func, CySimpleSlice)

if __name__ == "__main__":
    main()
