import timeit
import random
from ndindex import Tuple
from simple_tuple import SimpleTuple

from ndindex.simple_tuple_cython import SimpleTupleCython

from IPython.core.magics.execution import _format_time

tuple_classes = [Tuple, SimpleTuple, SimpleTupleCython]

N_RUNS = 1_000_000

def random_value():
    return random.choice([None] + list(range(-100, -1)) + list(range(1, 100)))

def generate_inputs(n=N_RUNS):
    return [(slice(random_value(), random_value(), random_value()),
             slice(random_value(), random_value(), random_value())) for _ in range(n)]

def benchmark_creation(TupleClass, inputs):
    for input in inputs:
        TupleClass(*input)

def benchmark_args(TupleClass, n=N_RUNS):
    s = TupleClass(slice(0, 10, 1), slice(100, 200, 1))
    for _ in range(n):
        s1, s2 = s.args

def benchmark_raw(TupleClass, n=N_RUNS):
    s = TupleClass(slice(0, 10, 1), slice(100, 200, 1))
    for _ in range(n):
        s.raw

def run_benchmark(name, func, TupleClass, *args):
    time = timeit.timeit(lambda: func(TupleClass, *args), number=1)
    print(f"{TupleClass.__name__:<30}: {_format_time(time/N_RUNS)}")

def main():
    print("Benchmarking SimpleTuple implementations")
    print("----------------------------------------")

    inputs = generate_inputs()

    print("\nObject Creation:")
    for TupleClass in tuple_classes:
        run_benchmark("Creation", benchmark_creation, TupleClass, inputs)

    print("\nArgs Access:")
    for TupleClass in tuple_classes:
        run_benchmark("Access", benchmark_args, TupleClass)

    print("\nRaw Access:")
    for TupleClass in tuple_classes:
        run_benchmark("Raw", benchmark_raw, TupleClass)

if __name__ == "__main__":
    main()
