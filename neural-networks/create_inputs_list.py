import os
import argparse
from pathlib import Path

from src.utils import common

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('images_dir', help='Path to directory containing the images')
    parser.add_argument('-o', '--output', default=None, help='Path to output file containing the list of images')
    args = parser.parse_args()

    absolute_imgs_dir = str(Path(args.images_dir).absolute())
    imgs_path_list = list(map(lambda img: f"{absolute_imgs_dir}/{img}", os.listdir(absolute_imgs_dir)))
    
    # Write to file
    default_out_file = f"{common.INPUTS_DIR}/{Path(args.images_dir).stem}.txt"
    out_file = args.output if not args.output is None else default_out_file
    imgs_path_as_str = "\n".join(imgs_path_list)

    with open(out_file, 'w') as f:
        f.write(imgs_path_as_str)

    f.close()

    print(f"Inputs file written to `{out_file}` with {len(imgs_path_list)} image(s)")

if __name__ == "__main__":
    main()