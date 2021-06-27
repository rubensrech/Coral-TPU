#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

from PIL import Image

from src.utils import common
from src.utils import detection

def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to directory with input images')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    args = parser.parse_args()

    labels = common.read_label_file(args.labels) if args.labels else {}

    cpu = not Path(args.model).stem.endswith('_edgetpu')
    interpreter = common.create_interpreter(args.model, cpu)
    interpreter.allocate_tensors()

    images_list = []
    total_imgs = 0

    for image_file in os.listdir(args.input):
        total_imgs += 1

        image_file_path = Path(args.input + '/' + image_file).absolute()
        image = Image.open(image_file_path)
        resized_image, scale = common.resize_input(image, interpreter)
        common.set_resized_input(interpreter, resized_image)

        interpreter.invoke()

        objs = detection.get_objects(interpreter, scale, args.threshold)

        print(f"> PREDICTING: {image_file}")

        if not objs:
            print("     No objects detected")
        else:
            images_list.append(image_file_path)

            for obj in objs:
                print("     " + labels.get(obj.id, obj.id))
                print("         id:    ", obj.id)
                print("         score: ", obj.score)
                print("         bbox:  ", obj.bbox)

    dataset_name = Path(args.input).stem
    out_file = dataset_name + '.txt'
    with open(out_file, 'w') as f:
        for img_path in images_list:
            f.write(str(img_path) + '\n')
        f.close()

    print(f"Detected images: {len(images_list)}/{total_imgs}")
    print(f"Images list saved to: {out_file}")


if __name__ == '__main__':
    main()
