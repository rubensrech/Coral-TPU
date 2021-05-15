import time
import argparse
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

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
                        help='File path of image to process')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    parser.add_argument('-d', '--draw', action='store_true', default=False,
                        help='Whether the detected boxes should be draw over the image')
    args = parser.parse_args()

    labels = common.read_label_file(args.labels) if args.labels else {}

    cpu = not Path(args.model).stem.endswith('_edgetpu')
    interpreter = common.create_interpreter(args.model, cpu)
    interpreter.allocate_tensors()

    image = Image.open(args.input)
    _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    print('----INFERENCE TIME----')
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    objs = detection.get_objects(interpreter, scale, args.threshold)
    print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    if args.draw:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.show()


if __name__ == '__main__':
    main()
