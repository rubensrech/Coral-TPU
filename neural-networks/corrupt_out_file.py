import os
import numpy as np
import argparse
from src.utils import common

ORIG_SUFFIX = ".orig"

def corrupt_out_file(out_file, ncorruptions=1):
    # Save original file
    os.system(f'cp -n {out_file} {out_file + ORIG_SUFFIX}')

    out_data = common.load_tensors_from_file(out_file)
    if 'detection_output' in out_data:
        tensorName = 'detection_output'
    elif 'scores' in out_data:
        tensorName = 'scores'
    else:
        raise Exception("Invalid file")

    for _ in range(ncorruptions):
        maxLinIdx = np.product(out_data[tensorName].shape)
        randLinPos = np.random.randint(maxLinIdx)
        if np.issubdtype(out_data[tensorName].dtype, np.floating):
            randVal = np.random.rand()
        else:
            randVal = np.random.randint(0, 255)
        prevVal = out_data[tensorName].ravel()[randLinPos]
        out_data[tensorName].ravel()[randLinPos] = randVal
        position = np.unravel_index(randLinPos, out_data[tensorName].shape)
        print(f"Corruption (position: {position}, prevVal: {prevVal}, newVal: {randVal})")

    np.save(out_file, out_data)

def revert(out_file):
    orig_file = out_file + ORIG_SUFFIX
    os.system(f'cp {orig_file} {out_file}')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out_file', help='Path to the file to be corrupted')
    parser.add_argument('-n', '--ncorruptions', type=int, default=1, help='Number of corruptions')
    parser.add_argument('-R', '--revert', action='store_true', help='Revert back to original file')
    args = parser.parse_args()

    if args.revert:
        revert(args.out_file)
    else:
        corrupt_out_file(args.out_file, args.ncorruptions)

if __name__ == '__main__':
    main()
