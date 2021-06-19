import os
import json
import argparse

from pycocotools.coco import COCO

def print_all_categories(coco):
    allCats = coco.loadCats(coco.getCatIds())
    print("Categories:")
    for cat in allCats:
        print(f"  - {cat['name']} (super category: {cat['supercategory']})")

def print_all_super_categories(coco):
    allCats = coco.loadCats(coco.getCatIds())
    allSupCats = set([cat['supercategory'] for cat in allCats])
    print("Super Categories:")
    for supCat in allSupCats:
        print("  - " + supCat)

def select_images_and_get_details(coco, supCatsNames, catNames, imgsPerCat):
    supCatsIds = coco.getCatIds(supNms=supCatsNames)
    catsIds = supCatsIds.expand(coco.getCatIds(catNms=catNames))
    catsIds = set(catsIds) # Remove duplicates

    imgIds = []
    for c in catsIds:
        cImgs = coco.getImgIds(catIds=c)
        imgIds.extend(cImgs if len(cImgs) < imgsPerCat else cImgs[0:imgsPerCat])

    imgIds = set(imgIds) # Remove duplicates
    return coco.loadImg(imgIds)

def write_output_files(imgDetails, imgsDir, cocoSet):
    out_files_prefix = f"subcoco_{cocoSet}_images_"
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
    parser.add_argument('--coco_dir', required=True, default="/mnt/4E0AEF320AEF15AD/RUBENS/det_out/coco",
                help='Root directory to raw Microsoft COCO dataset.')
    parser.add_argument('--set', required=True, default="val2017",
                help='Select training or validation set.')
    parser.add_argument('--images_per_category', required=False, default=20, type=bool,
                help='Number of images of each category to be included in the output list.')
    parser.add_argument('--super_categories', required=False, default="person,vehicle,outdoor",
                help='Comma-separated list of COCO super categories.')
    parser.add_argument('--categories', required=False, default=None,
                help='Comma-separated list of COCO categories. This will be merged to `super_categories`.')
    parser.add_argument('--print_categories', required=False, default=False, type=bool,
                help='Set this flag to print all supported COCO categories/super categories.')
    args = parser.parse_args()

    annotationsFilePath = os.path.join(args.coco_dir, 'annotations', f'instances_{args.set}.json')
    imgsDir = os.path.join(args.coco_dir, args.set)

    coco = COCO(annotationsFilePath)

    if args.print_categories:
        print_all_super_categories()
        print("") # Empty line
        print_all_categories()
        exit(0)

    supCatsNames = args.super_categories.split(',') if args.super_categories else []
    catsNames = args.categories.split(',') if args.categories else []

    imgDetails = select_images_and_get_details(coco, supCatsNames, catsNames, args.images_per_category)
    write_output_files(imgDetails, imgsDir, args.set)


if __name__ == "__main__":
    main()