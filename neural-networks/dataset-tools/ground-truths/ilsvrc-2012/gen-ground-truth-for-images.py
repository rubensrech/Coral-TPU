import re
import csv
import argparse

from typing import Dict, List
from pathlib import Path

IMAGE_NAME_PATTERN = re.compile(r"ILSVRC2012_val_(\d+)")
GROUND_TRUTH_MAP_FILE_LINE_PATTERN = re.compile(r"ILSVRC2012_val_(\d+)\.[a-zA-Z]+\s+(\d+)")
GROUND_TRUTH_MAP_CLASS_IDX_OFFSET = 1

def load_labels_map(labels_file: str):
    with open(labels_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    labels_map = {}

    for i, line in enumerate(lines):
        pair = re.split(r'[:\s]+', line.strip(), maxsplit=1)
        assert len(pair) == 2, f"Invalid line [{i}] in file [{labels_file}]"

        class_id = int(pair[0])
        class_label = pair[1].strip()
        labels_map[class_id] = class_label

    return labels_map

def load_ground_truth_map(map_file_path: str):
    with open(map_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    ground_truth_map = {}

    for i, line in enumerate(lines):
        match = GROUND_TRUTH_MAP_FILE_LINE_PATTERN.match(line)
        assert match is not None, f"Invalid line [{i}] in file [{map_file_path}] (#1)"
        assert len(match.groups()) == 2, f"Invalid line [{i}] in file [{map_file_path}] (#2)"

        img_id = int(match.group(1).strip())
        class_id = int(match.group(2).strip())
        ground_truth_map[img_id] = class_id

    return ground_truth_map
        
def gen_output_mapping(imgs_file: str, ground_truth_map: Dict[int, int], labels_map: Dict[int, str]={}):
    with open(imgs_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_mapping = []
    
    for i, img_path in enumerate(lines):
        img_name = Path(img_path).stem
        match = IMAGE_NAME_PATTERN.match(img_name)
        assert match is not None, f"Invalid line [{i}] in file [{imgs_file}] (#1)"
        assert len(match.groups()) == 1, f"Invalid line [{i}] in file [{imgs_file}] (#2)"

        img_id = int(match.group(1).strip())
        class_id = ground_truth_map[img_id] + GROUND_TRUTH_MAP_CLASS_IDX_OFFSET
        class_label = labels_map[class_id]

        out_mapping.append({
            "img_id": img_id,
            "img_path": img_path.strip(),
            "img_name": img_name,
            "class_id": class_id,
            "class_label": class_label
        })
    
    return out_mapping
    
def write_output_file(out_file: str, out_mapping: List[Dict[str, any]], header: List[str]):
    with open(out_file, "w") as f:
        writer = csv.writer(f)
    
        writer.writerow(header)

        for img_info in out_mapping:
            row = [img_info[column] for column in header]
            writer.writerow(row)

        f.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--imgs_file', required=True,
                        help='Path to TXT file containing a list of images from ILSVRC2012 dataset.')
    parser.add_argument('-O', '--output_file', default="imagenet-subset-ground-truths.csv",
                        help='Path to CSV output file.')
    parser.add_argument('-M', '--ground_truth_map_file', default="ground-truth-map.txt",
                        help='Path to TXT file that maps the image name to its respective ground truth class.')
    parser.add_argument('-H', '--header', default="img_name,img_path,img_id,class_id,class_label",
                        help='The header columns that will be included in the output CSV file.')
    parser.add_argument('--labels_file', default="../../../labels/imagenet_labels.txt",
                        help='''Whether the output CSV file will include a column indicating the human-readable
                            value of the ground truth class id.''')
    args = parser.parse_args()

    header_columns = args.header.split(",")

    ground_truth_map = load_ground_truth_map(args.ground_truth_map_file)
    labels_map = load_labels_map(args.labels_file) if "class_label" in header_columns else {}
    out_mapping = gen_output_mapping(args.imgs_file, ground_truth_map, labels_map)
    write_output_file(args.output_file, out_mapping, header_columns)

    print(f"Successfully wrote ground truth info for [{len(out_mapping)}] images to file [{args.output_file}]")

if __name__ == "__main__":
    main()