#!/usr/bin/env python3

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(1, './src')
import util
from util import Operation

np.random.seed(0)

INPUTS_DIR = f"{Path(__file__).parent}/inputs"

def get_input_image_name(image_filename):
    return Path(image_filename).stem

def get_output_name(model_file, out_ext, image_file=None):
    if image_file is not None:
        out_desc = get_input_image_name(image_file)
    else:
        out_desc = "rand"

    op = util.get_op_from_model_name(model_file)
    opStr = op.value
    dims = util.get_dims_from_model_name(model_file)
    dimsStr = "_".join(map(str, dims))
    
    return f"{INPUTS_DIR}/{out_desc}-{op.value}_{dimsStr}.{out_ext}"

def generate_from_input_image(model_file, image_file, output_size, op, out_ext):
    img = Image.open(image_file)
    img = img.resize(output_size[0:2])

    if op == Operation.Conv2d:
        img = img.convert('L')

    output_file = get_output_name(model_file, out_ext, image_file)
    output_arr = np.array(img)
    output_shape = output_arr.shape
    
    if out_ext == "bmp":
        img.save(output_file)
    elif out_ext == "npy":
        np.save(output_file, output_arr)

    print(f'Output image saved to `{output_file}` with dimensions {output_shape}')

def generate_random_input(model_file, size, op, out_ext):
    rand_input = np.random.randint(0, 255, size, dtype=np.uint8)
    output_file = get_output_name(model_file, out_ext)

    if out_ext == "bmp":
        Image.fromarray(np.squeeze(rand_input)).save(output_file)
    elif out_ext == "npy":
        np.save(output_file, rand_input)

    zero_count = np.sum(rand_input == 0)
    print(f'Random input saved to `{output_file}` with dimensions {size} and {zero_count} zero element(s)')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input', required=False, default=None,
                        help='Path to the input image (.jpg,.png,...)\
                              If not available, random input will be generated')
    parser.add_argument('-M', '--model', required=True,
                        help='Path to the model file that will receive the BMP image as input')
    parser.add_argument('--npy', required=False, action='store_true', default=False,
                        help='Wether the output file should be a .NPY (Numpy file)')             
    args = parser.parse_args()

    model_file = args.model
    image_file = args.input
    npy_out = args.npy

    op = util.get_op_from_model_name(model_file)
    dims = util.get_dims_from_model_name(model_file)

    output_size = dims[1:4]
    output_ext = "npy" if npy_out else "bmp"

    if image_file is not None:
        generate_from_input_image(model_file, image_file, output_size, op, output_ext)
    else:
        generate_random_input(model_file, output_size, op, output_ext)

if __name__ == "__main__":
    main()