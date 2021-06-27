#!/usr/bin/env python3

import argparse
from src.utils.common import Dataset

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('images_dir', help='Path to directory containing the images')
    parser.add_argument('-n', '--nimages', type=int, default=None, help='Max number of images that should be included')
    args = parser.parse_args()

    dataset = Dataset(args.images_dir, args.nimages)
    out_file, nimages = dataset.create_input_images_file()

    print(f"Inputs file written to `{out_file}` with {nimages} image(s)")

if __name__ == "__main__":
    main()