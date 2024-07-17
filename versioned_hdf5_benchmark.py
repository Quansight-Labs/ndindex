import h5py
from versioned_hdf5 import VersionedHDF5File
import numpy as np

import argparse


def setup(d):
    with h5py.File(d, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            sv.create_dataset('values', data=np.arange(100_000_000), chunks=(1_000,), maxshape=(None,))

def resize(d):
    with h5py.File(d, mode="r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r1") as sv:
            values = sv['values']
            # resizing creates an InMemoryDataset and populates data_dict
            values.resize((110_000_000,))
            # reading from InMemoryDataset is now slow
            _ = values[500:90_000_000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='./data.h5')
    parser.add_argument('--setup', action='store_true')

    args = parser.parse_args()

    if args.setup:
        setup(args.filename)
    else:
        resize(args.filename)

if __name__ == '__main__':
    main()
