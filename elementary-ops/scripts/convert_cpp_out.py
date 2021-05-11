import sys
import numpy as np
from PIL import Image

filename = sys.argv[1]

with open(filename, 'rb') as f:
    # Read .out file
    dimsSize = int.from_bytes(f.read(4), 'little')
    dims = np.fromfile(f, dtype=np.int32, count=4)
    n = int.from_bytes(f.read(8), 'little')
    assert np.prod(dims) == n
    data = np.fromfile(f, dtype=np.uint8).reshape(dims)

    # Save to .bmp file
    Image.fromarray(np.squeeze(data)).save(filename + '.bmp')
    # Save to .npy file
    np.save(filename + '.npy', data)