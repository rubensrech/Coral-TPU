#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from src import util
from src.util import Operation

np.random.seed(0)

def generate_from_input_image(model: str, out_ext: str, image_file: str):
    op, input_shape, _, _ = util.parse_model_name(model)

    img = Image.open(image_file)
    img = img.resize(input_shape[1:3])

    if op == Operation.Conv2d:
        img = img.convert('L')

    output_file = util.generate_input_filename(model, out_ext, image_file)
    output_arr = np.array(img)
    
    if out_ext == "bmp":
        img.save(output_file)
    elif out_ext == "npy":
        np.save(output_file, output_arr)

    return output_file, output_arr
    
def generate_random_input(model: str, out_ext: str):
    input_shape, _ = util.get_dims_from_model_name(model)
    output_size = input_shape[1:4]

    rand_input = np.random.randint(0, 255, output_size, dtype=np.uint8)
    output_file = util.generate_input_filename(model, out_ext)

    if out_ext == "bmp":
        Image.fromarray(np.squeeze(rand_input)).save(output_file)
    elif out_ext == "npy":
        np.save(output_file, rand_input)

    return output_file, rand_input

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

    output_ext = "npy" if npy_out else "bmp"

    if image_file:
        out_file, out_arr = generate_from_input_image(model_file, output_ext, image_file)
    else:
        out_file, out_arr = generate_random_input(model_file, output_ext)

    zero_count = np.sum(out_arr == 0)
    print(f'Generated input saved to `{out_file}` with dimensions {out_arr.shape} and {zero_count}')

if __name__ == "__main__":
    main()
