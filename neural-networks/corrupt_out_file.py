#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
from src.utils import common

ORIG_SUFFIX = ".orig"

def corrupt_out_file(out_file, ncorruptions=1):
    # Save original file
    os.system(f'cp -n {out_file} {out_file + ORIG_SUFFIX}')

    out_data = common.load_tensors_from_file(out_file)
    if 'detection_output' in out_data: # Detection
        tensorName = 'detection_output'
    elif 'scores' in out_data: # Classification
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
    parser.add_argument('-r', '--revert_delay_ms', type=int, default=500, help='The delay, in ms, before undoing the corruption and reverting back to original file')
    args = parser.parse_args()

    revert_delay_seconds = args.revert_delay_ms / 1000

    corrupt_out_file(args.out_file, args.ncorruptions)

    if revert_delay_seconds > 0:
        time.sleep(revert_delay_seconds)
        revert(args.out_file)

if __name__ == '__main__':
    main()
