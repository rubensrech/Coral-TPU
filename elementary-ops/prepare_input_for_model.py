
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

def get_output_image_name(model_file, image_file):
    img_name = get_input_image_name(image_file)
    op = util.get_op_from_model_name(model_file)
    opStr = op.value
    dims = util.get_dims_from_model_name(model_file)
    dimsStr = "_".join(map(str, dims))
    ext = "bmp"
    return f"{INPUTS_DIR}/{img_name}-{op.value}_{dimsStr}.{ext}"

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input', required=True,
                        help='Path to the input image (.jpg,.png,...)')
    parser.add_argument('-M', '--model', required=True,
                        help='Path to the model file that will receive the BMP image as input')
    args = parser.parse_args()

    model_file = args.model
    image_file = args.input

    op = util.get_op_from_model_name(model_file)
    dims = util.get_dims_from_model_name(model_file)

    outImgSize = dims[1:3]

    img = Image.open(image_file)
    img = img.resize(outImgSize)

    if op == Operation.Conv2d:
        img = img.convert('L')

    outImgShape = np.array(img).shape
    outImgFile = get_output_image_name(model_file, image_file)

    img.save(outImgFile)

    print(f'Output image saved to `{outImgFile}` with dimensions {outImgShape}')

if __name__ == "__main__":
    main()