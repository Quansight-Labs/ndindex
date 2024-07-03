import random
import time
import sys

from ndindex import Slice
from simple_slice import SimpleSlice
from simple_slice_cython import SimpleSlice as SimpleSliceCython

sys.path.append('/Users/aaronmeurer/Documents/mypython')
from mypython.timeit import timeit_format

N_RUNS = 1000000

def time_slice(SliceClass):
    x, y, z = random.choice([None, -1, 0, 1, 2]), random.choice([None, -1, 0, 1, 2]), random.choice([None, -1, 1, 2])
    t = time.perf_counter()
    SliceClass(x, y, z)
    return time.perf_counter() - t

if __name__ == '__main__':
    times = [time_slice(Slice) for i in range(N_RUNS)]

    simple_times = [time_slice(SimpleSlice) for i in range(N_RUNS)]

    cython_times = [time_slice(SimpleSliceCython) for i in range(N_RUNS)]

    print(f"Slice times ({N_RUNS} runs):")
    print(timeit_format(times, 'Slice times'))
    print()

    print(f"SimpleSlice times ({N_RUNS} runs):")
    print(timeit_format(simple_times, 'SimpleSlice times'))
    print()

    print(f"SimpleSlice Cython times ({N_RUNS} runs):")
    print(timeit_format(cython_times, 'SimpleSlice Cython times'))
