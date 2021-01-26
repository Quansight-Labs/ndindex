import numpy as np
from ndindex import ndindex

class TimeEquality():
    def setup(self):
        self.builtin_types = [0, np.int64(0), [0, 1], True, False,
                              np.array([0, 1]), np.array([True, False]),
                              np.array(True), np.bool_(True), np.array(0),
                              ..., slice(0, 1), None, (slice(0, 1), ..., 0)]
        self.ndindex_types = [ndindex(i) for i in self.builtin_types]

    def time_equality_ndindex_builtin(self):
        for ndindex_idx in self.ndindex_types:
            for builtin_idx in self.builtin_types:
                ndindex_idx == builtin_idx

    def time_equality_ndindex_ndindex(self):
        for ndindex_idx1 in self.ndindex_types:
            for ndindex_idx2 in self.ndindex_types:
                ndindex_idx1 == ndindex_idx2

    def time_hash(self):
        for ndindex_idx in self.ndindex_types:
            hash(ndindex_idx)
