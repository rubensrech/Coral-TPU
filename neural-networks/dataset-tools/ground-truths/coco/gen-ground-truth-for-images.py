import re
import csv
import argparse
from typing import Dict, List

from pathlib import Path
from pycocotools.coco import COCO

IMAGE_NAME_PATTERN = re.compile(r"(\d+)")

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

def getAnnotationsForImage(img_id: str, coco: COCO, labels_map: Dict[int, str]={}):
    ann_ids = coco.getAnnIds(img_id)
    anns = coco.loadAnns(ann_ids)

    objs_data = []

    for ann in anns:
        objs_data.append({
            "class_id": ann["category_id"],
            "bbox_x": ann["bbox"][0],
            "bbox_y": ann["bbox"][1],
            "bbox_w": ann["bbox"][2],
            "bbox_h": ann["bbox"][3],
            "class_label": labels_map[ann["category_id"]] if ann["category_id"] in labels_map else None
        })
    
    return objs_data

def gen_output_mapping(imgs_file: str, anns_file: str, labels_map: Dict[int, str]={}):
    with open(imgs_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    coco = COCO(anns_file)
    out_mapping = []
    
    for i, img_path in enumerate(lines):
        img_name = Path(img_path).stem
        match = IMAGE_NAME_PATTERN.match(img_name)
        assert match is not None, f"Invalid line [{i}] in file [{imgs_file}] (#1)"
        assert len(match.groups()) == 1, f"Invalid line [{i}] in file [{imgs_file}] (#2)"

        img_id = int(match.group(1).strip())
        annotations = getAnnotationsForImage(img_id, coco, labels_map)

        out_mapping.append({
            "img_id": img_id,
            "img_path": img_path.strip(),
            "img_name": img_name,
            "annotations": annotations
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
                        help='Path to TXT file containing a list of images from COCO dataset.')
    parser.add_argument('-A', '--coco_annotations_file', default="/Users/rubensrechjunior/Downloads/TPU/coco2017_train_val_annotations/instances_val2017.json",
                        help='Path to JSON file containing COCO annotations.')
    parser.add_argument('-O', '--output_file', default="coco-subset-ground-truths.csv",
                        help='Path to CSV output file.')
    parser.add_argument('-H', '--header', default="img_name,img_path,annotations",
                        help='The header columns that will be included in the output CSV file.')
    parser.add_argument('--labels_file', default="../../../labels/coco_labels.txt",
                        help='''Whether the output CSV file will include a column indicating the human-readable
                            value of the ground truth class id.''')
    args = parser.parse_args()

    header_columns = args.header.split(",")

    labels_map = load_labels_map(args.labels_file)
    out_mapping = gen_output_mapping(args.imgs_file, args.coco_annotations_file, labels_map)
    write_output_file(args.output_file, out_mapping, header_columns)

    print(f"Successfully wrote ground truth info for [{len(out_mapping)}] images to file [{args.output_file}]")

if __name__ == "__main__":
    main()