#!/usr/bin/env python3

import time
import argparse
from pathlib import Path

from PIL import Image

from src.utils import common
from src.utils import classification

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of image to process')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for classification scores')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Top K classes')
    args = parser.parse_args()

    labels = common.read_label_file(args.labels) if args.labels else {}

    cpu = not Path(args.model).stem.endswith('_edgetpu')
    interpreter = common.create_interpreter(args.model, cpu)
    interpreter.allocate_tensors()

    image = Image.open(args.input)
    resized_image, scale = common.resize_input(image, interpreter)
    common.set_resized_input(interpreter, resized_image)

    print('----INFERENCE TIME----')
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classification.get_classes(interpreter, args.topk, args.threshold)
    print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

if __name__ == '__main__':
    main()
