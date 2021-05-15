#!/usr/local/bin/python3

import time
import argparse
from typing import IO
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils.logger import Logger
Logger.setLevel(Logger.Level.DEBUG)

from src.utils import common

MAX_ERR_PER_IT = 500
INCLUDE_CORAL_OUT_IN_SDC_FILES = False

# Auxiliary functions

def save_output_to_file(raw_out_dict, filename, model_input_size, input_image_scale):
    data = { **raw_out_dict, 'input_image_scale': input_image_scale, 'model_input_size': model_input_size }
    common.save_tensors_to_file(data, filename)

# Main functions

def create_interpreter(model_file, cpu=False):
    t0 = time.perf_counter()

    interpreter = common.create_interpreter(model_file, cpu)

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

def save_golden_output(interpreter, model_file, image_file, img_scale, coral_out_tensors):
    t0 = time.perf_counter()

    golden_file = common.get_dft_golden_filename(model_file, image_file)
    raw_out = common.get_raw_output(interpreter, coral_out_tensors)
    model_in_size = common.input_size(interpreter)
    save_output_to_file(raw_out, golden_file, model_in_size, img_scale)

    t1 = time.perf_counter()

    Logger.info(f"Golden output saved to file `{golden_file}`")
    Logger.timing("Save golden output", t1 - t0)

    return golden_file

def check_output_against_golden(interpreter, coral_out_tensors, golden_file):
    t0 = time.perf_counter()

    try:
        gold_out = common.load_tensors_from_file(golden_file)
        curr_out = common.get_raw_output(interpreter, coral_out_tensors)
    except IOError:
        raise Exception("Could not open golden file")

    out_total_errs_count = 0
    logged_errs_count = 0

    for tensorKey in curr_out:
        try:
            gold_tensor = gold_out[tensorKey]
            curr_tensor = curr_out[tensorKey]

            if type(curr_tensor) is int:
                gold_tensor = np.array(gold_tensor)
                curr_tensor = np.array(curr_tensor)

            if curr_tensor.shape != gold_tensor.shape:
                raise Exception("Invalid golden file for current execution")

            diff_pos = np.flatnonzero(gold_tensor != curr_tensor)
            tensor_errs_count = len(diff_pos)
            if tensor_errs_count > 0:
                out_total_errs_count += tensor_errs_count

                expected = gold_tensor.ravel()[diff_pos]
                result = curr_tensor.ravel()[diff_pos]

                for i in range(tensor_errs_count):
                    if logged_errs_count < MAX_ERR_PER_IT:
                        pos = np.unravel_index(diff_pos[i], curr_tensor.shape)
                        Logger.error(f"tensor: {tensorKey}, position: {pos}, expected: {expected[i]}, result: {result[i]}")
                        logged_errs_count += 1

        except KeyError as ex:
            raise Exception("Invalid golden file for current execution") from ex

    t1 = time.perf_counter()

    if out_total_errs_count > 0:
        Logger.error(f"Output doesn't match golden: {out_total_errs_count} error(s)")
    else:
        Logger.info(f"Output matches golden")

    Logger.timing("Check output", t1 - t0)
            
    return out_total_errs_count

def save_sdc_output(interpreter, model_file, img_file, img_scale, coral_out_tensors=[]):
    t0 = time.perf_counter()

    sdc_out_file = common.get_dft_sdc_out_filename(model_file, img_file)
    raw_out = common.get_raw_output(interpreter, coral_out_tensors if INCLUDE_CORAL_OUT_IN_SDC_FILES else [])
    model_in_size = common.input_size(interpreter)
    save_output_to_file(raw_out, sdc_out_file, model_in_size, img_scale)

    t1 = time.perf_counter()

    Logger.info(f"SDC output saved to file `{sdc_out_file}`")
    Logger.timing("Save SDC output", t1 - t0)

    return sdc_out_file

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('-m', '--model', required=True, help='File path to .tflite file')
    parser.add_argument('-i', '--input', required=True, help='File path to list of images to be processed')
    parser.add_argument('-t', '--coral-tensors', required=True, help='Tensor indexes of the output from Coral TPU (comma separated)')
    # Optionals
    parser.add_argument('--iterations', type=int, default=1, help='Number of times to run inference')
    parser.add_argument('--save-golden', action='store_true', default=False, help='Whether the output should be saved to a binary file in .npy format or not')
    args = parser.parse_args()

    model_file = args.model
    input_file = args.input
    coral_out_tensors = list(map(int, args.coral_tensors.split(",")))
    iterations = args.iterations
    save_golden = args.save_golden

    cpu = not Path(model_file).stem.endswith('_edgetpu')

    interpreter = create_interpreter(model_file, cpu)

    for i in range(iterations):
        Logger.info(f"Iteration {i}")

        with open(input_file, 'r') as f:
            inputs = f.read().splitlines()

        for image_file in inputs:
            Logger.info(f"Predicting image: {image_file}")

            _, img_scale = set_interpreter_intput(interpreter, image_file)

            perform_inference(interpreter)

            if save_golden:
                save_golden_output(interpreter, model_file, image_file, img_scale, coral_out_tensors)
            else:
                golden_file = common.get_dft_golden_filename(model_file, image_file)
                errs_count = check_output_against_golden(interpreter, coral_out_tensors, golden_file)
                if errs_count > 0:
                    sdc_file = save_sdc_output(interpreter, model_file, image_file, img_scale, coral_out_tensors)
                    Logger.info(f"SDC output saved to file `{sdc_file}`")

                    Logger.error(f"SDC: {errs_count} error(s) detected")
                    # LogHelper.log_error_count(errs_count)

        if save_golden:
            break

if __name__ == '__main__':
    main()
