#!/usr/local/bin/python3

import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from enum import Enum

from src.utils.logger import Logger
Logger.setLevel(Logger.Level.DEBUG)

from src.utils import common
from src.utils import detection

# Auxiliary functions

def save_raw_output_to_file(interpreter, filename):
    detection.get_raw_output(interpreter).save_to_file(filename)

# Main functions

def create_interpreter(model_file, cpu=False):
    t0 = time.perf_counter()

    if cpu:
        from tensorflow import lite as tflite
        interpreter = tflite.Interpreter(model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter        
        interpreter = make_interpreter(model_file)

    interpreter.allocate_tensors()

    t1 = time.perf_counter()

    Logger.info("Interpreter created successfully")
    Logger.timing("Create interpreter", t1 - t0)

    return interpreter

def set_interpreter_intput(interpreter, image_file):
    t0 = time.perf_counter()

    image = Image.open(image_file)
    in_tensor, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS)) 

    t1 = time.perf_counter()

    Logger.info("Interpreter input set successfully")
    Logger.timing("Set interpreter input", t1 - t0)

    return in_tensor, scale

def perform_inference(interpreter):
    t0 = time.perf_counter()

    interpreter.invoke()

    t1 = time.perf_counter()

    Logger.info("Inference performed successfully")
    Logger.timing("Perform inference", t1 - t0)

def save_golden_output(interpreter, model_file, image_file):
    t0 = time.perf_counter()

    golden_file = common.get_dft_golden_filename(model_file, image_file)
    save_raw_output_to_file(interpreter, golden_file)

    t1 = time.perf_counter()

    Logger.info(f"Golden output saved to file `{golden_file}`")
    Logger.timing("Save golden output", t1 - t0)

def check_output_against_golden(interpreter, golden_file):
    t0 = time.perf_counter()

    gold_out = detection.DetectionRawOutput.from_file(golden_file)
    curr_out = detection.get_raw_output(interpreter)

    total_errs_count = 0

    for tensorName in gold_out._fields:
        gold_tensor = getattr(gold_out, tensorName)
        out_tensor = getattr(curr_out, tensorName)

        diffIdxs = np.flatnonzero(gold_tensor != out_tensor)
        tensor_errs_count = len(diffIdxs)
        if tensor_errs_count > 0:
            total_errs_count += tensor_errs_count

            expected = gold_tensor.ravel()[diffIdxs]
            result = out_tensor.ravel()[diffIdxs]
            for i in range(tensor_errs_count):
                Logger.error(f"tensor: {tensorName}, position: {diffIdxs[i]}, expected: {expected[i]}, result: {result[i]}")

    t1 = time.perf_counter()

    if total_errs_count > 0:
        Logger.error(f"Output doesn't match golden: {total_errs_count} error(s)")
    else:
        Logger.info(f"Output matches golden")

    Logger.timing("Check output", t1 - t0)
            
    return total_errs_count

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path to .tflite file')
    parser.add_argument('-i', '--input', required=True,
                        help='File path to list of images to be processed')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of times to run inference')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Whether the inference should be performed in the CPU or in the Edge TPU')
    parser.add_argument('--save-golden', action='store_true', default=False,
                        help='Whether the output should be saved to a binary file in .npy format or not')
    args = parser.parse_args()

    model_file = args.model
    input_file = args.input
    labels_file = args.labels
    iterations = args.iterations
    threshold = args.threshold
    cpu = args.cpu
    save_golden = args.save_golden

    labels = common.read_label_file(labels_file) if labels_file else {}

    interpreter = create_interpreter(model_file, cpu)

    for i in range(iterations):
        Logger.info(f"Iteration {i}")

        with open(input_file, 'r') as f:
            inputs = f.read().splitlines()

        for image_file in inputs:
            Logger.info(f"Predicting image: {image_file}")

            in_tensor, img_scale = set_interpreter_intput(interpreter, image_file)

            perform_inference(interpreter)

            if save_golden:
                save_golden_output(interpreter, model_file, image_file)
            else:
                golden_file = common.get_dft_golden_filename(model_file, image_file)
                check_output_against_golden(interpreter, golden_file)

        if save_golden:
            break



if __name__ == '__main__':
    main()
