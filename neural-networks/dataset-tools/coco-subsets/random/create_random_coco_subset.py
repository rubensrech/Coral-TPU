import os
import re
import json
import random 
import argparse

from pycocotools.coco import COCO

IMAGE_LIST_FILE_LINE_PATTERN = re.compile(r".*(\d{12})\.[a-zA-Z]+")

def get_image_details_for_images_in_file(image_list_file: str, coco: COCO):
    with open(image_list_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    img_details = []

    for i, line in enumerate(lines):
        match = IMAGE_LIST_FILE_LINE_PATTERN.match(line)
        assert match is not None, f"Invalid line [{i}] in file [{image_list_file}] (#1)"
        assert len(match.groups()) == 1, f"Invalid line [{i}] in file [{image_list_file}] (#2)"

        img_id = int(match.group(1).strip())
        img_details.append(coco.imgs[img_id])

    return img_details

def get_image_details_for_random_sample(coco: COCO, numImages: int):
    imgIndexes = random.sample(list(coco.imgs), numImages)
    imgDetails = [coco.imgs[idx] for idx in imgIndexes]

    return imgDetails

def write_output_files(imgDetails, imgsDir):
    out_files_prefix = f"rand_coco_subset_{len(imgDetails)}_image_"
    detailsFileName = out_files_prefix + "details.json"
    urlsFileName = out_files_prefix + "urls.txt"
    pathsFileName = out_files_prefix + "paths.txt"

    # Write details
    with open(detailsFileName, "w") as detailsFile:
        json.dump(imgDetails, detailsFile)
        detailsFile.close()
    
    # Write URLs
    with open(urlsFileName, "w") as urlsFile:
        for d in imgDetails:
            urlsFile.write(d["coco_url"] + "\n")
        urlsFile.close()

    # Write paths
    with open(pathsFileName, "w") as pathsFile:
        for d in imgDetails:
            path = os.path.join(imgsDir, d["file_name"])
            pathsFile.write(str(path) + "\n")
        urlsFile.close()
    
    print("Output files writen to:")
    print(f"  - Image details: {detailsFileName}")
    print(f"  - Image URLs: {urlsFileName}")
    print(f"  - Image paths: {pathsFileName}")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations_file_path', required=False, default="/Users/rubensrechjunior/Downloads/TPU/coco2017_train_val_annotations/instances_val2017.json",
                help='Path to the COCO annotations file. (Can be downloaded here: http://images.cocodataset.org/annotations/annotations_trainval2017.zip)')
    parser.add_argument('--num_images', required=False, type=int, default=100,
                help='Number of images to be included in the output list.')
    parser.add_argument('--images_dir', required=False, default="",
                help='Path to directory containing the COCO images.')
    parser.add_argument('--images_list_file', required=False, default=None,
                help='Path to file containing a list of images from COCO dataset. If this is not provided, a random sample of images will be used.')
    args = parser.parse_args()

    coco = COCO(args.annotations_file_path)
    imgDetails = get_image_details_for_images_in_file(args.images_list_file, coco) if args.images_list_file else get_image_details_for_random_sample(coco, args.num_images)
    write_output_files(imgDetails, args.images_dir)


if __name__ == "__main__":
    main()