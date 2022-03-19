import os
import json
import argparse
import random 

from pycocotools.coco import COCO

def select_images_and_get_details(coco: COCO, numImages: int):
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
    parser.add_argument('--annotations_file_path', required=True, default="/Users/rubensrechjunior/TPU/coco2017_train_val_annotations/instances_val2017.json",
                help='Path to the COCO annotations file. (Can be downloaded here: http://images.cocodataset.org/annotations/annotations_trainval2017.zip)')
    parser.add_argument('--num_images', required=True, type=int, default=100,
                help='Number of images to be included in the output list.')
    parser.add_argument('--images_dir', required=False, default="",
                help='Path to directory containing the COCO images.')
    args = parser.parse_args()

    coco = COCO(args.annotations_file_path)
    imgDetails = select_images_and_get_details(coco, args.num_images)
    write_output_files(imgDetails, args.images_dir)


if __name__ == "__main__":
    main()