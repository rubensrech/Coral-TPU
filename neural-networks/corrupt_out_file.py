import os
import numpy as np
import argparse
from src.utils import common

ORIG_SUFFIX = ".orig"

def corrupt_out_file(out_file, error_descriptors):
    # Save original file
    os.system(f'cp {out_file} {out_file + ORIG_SUFFIX}')

    out_data = common.load_tensors_from_file(out_file)

    for err in error_descriptors:
        position = err.get('position')
        count = err.get('count', 1)
        tensorName = err.get('tensor')
        
        if type(out_data[tensorName]) is int:
            randVal = np.random.randint(0, 255)
            prevVal = out_data[err.get('tensor')]
            out_data[err.get('tensor')] = randVal
            print(f"Corruption (tensor: {err.get('tensor')}, position: 0, prevVal: {prevVal}, newVal: {randVal})")
        else:
            for i in range(count):
                maxLinIdx = np.product(out_data[tensorName].shape)
                randLinPos = np.random.randint(maxLinIdx)
                randVal = randVal = np.random.rand() if out_data[tensorName].dtype == 'float32' else np.random.randint(0, 255)
                prevVal = out_data[tensorName].ravel()[randLinPos]
                out_data[tensorName].ravel()[randLinPos] = randVal
                position = np.unravel_index(randLinPos, out_data[tensorName].shape)
                print(f"Corruption (tensor: {err.get('tensor')}, position: {position}, prevVal: {prevVal}, newVal: {randVal})")

    np.save(out_file, out_data)

def revert(out_file):
    orig_file = out_file + ORIG_SUFFIX
    os.system(f'cp {orig_file} {out_file}', )
    os.unlink(orig_file)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out_file', help='Path to the file to be corrupted')
    parser.add_argument('-R', '--revert', action='store_true', help='Revert back to original file')
    args = parser.parse_args()

    if args.revert:
        revert(args.out_file)
    else:
        corrupt_out_file(args.out_file, [
            { 'tensor': 'boxes', 'count': 1 },
            # { 'tensor': 'class_ids', 'count': 1 },
            # { 'tensor': 'scores', 'count': 1 },
            # { 'tensor': 'count', 'count': 1 },
        ])

if __name__ == '__main__':
    main()
