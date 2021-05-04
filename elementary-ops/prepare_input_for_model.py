#!/usr/bin/env python3

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(1, './src')
import util
from util import Operation

INPUTS_DIR = f"{Path(__file__).parent}/inputs"

def get_input_image_name(image_filename):
    return Path(image_filename).stem

def get_output_name(model_file, image_file=None):
    if image_file is not None:
        out_desc = get_input_image_name(image_file)
        ext = ".bmp"
    else:
        out_desc = "rand"
        ext = ".npy"

    op = util.get_op_from_model_name(model_file)
    opStr = op.value
    dims = util.get_dims_from_model_name(model_file)
    dimsStr = "_".join(map(str, dims))
    
    return f"{INPUTS_DIR}/{out_desc}-{op.value}_{dimsStr}{ext}"

def generate_from_input_image(model_file, image_file, output_size, op):
    img = Image.open(image_file)
    img = img.resize(output_size)

    if op == Operation.Conv2d:
        img = img.convert('L')

    output_shape = np.array(img).shape
    output_file = get_output_name(model_file, image_file)

    img.save(output_file)

    print(f'Output image saved to `{output_file}` with dimensions {output_shape}')

def generate_random_input(model_file, size, op):
    rand_input = np.random.randint(0, 255, size, dtype=np.uint8)
    output_file = get_output_name(model_file)
    np.save(output_file, rand_input)
    print(f'Random input saved to `{output_file}` with dimensions {size}')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input', required=False, default=None,
                        help='Path to the input image (.jpg,.png,...)\
                              If not available, random input will be generated')
    parser.add_argument('-M', '--model', required=True,
                        help='Path to the model file that will receive the BMP image as input')
    args = parser.parse_args()

    model_file = args.model
    image_file = args.input

    op = util.get_op_from_model_name(model_file)
    dims = util.get_dims_from_model_name(model_file)

    output_size = dims[1:3]

    if image_file is not None:
        generate_from_input_image(image_file, output_size, op)
    else:
        generate_random_input(model_file, output_size, op)

if __name__ == "__main__":
    main()