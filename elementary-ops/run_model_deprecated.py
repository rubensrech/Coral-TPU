#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from enum import Enum
from time import time
from PIL import Image
from pathlib import Path
from pycoral.adapters import common

import sys
# import util
sys.path.insert(1, './src')
import util
from util import Operation, Plataform

# import _log_helper
sys.path.insert(0, './include/log_helper_swig_wraper/')
import _log_helper as lh

GOLDEN_DIR = f"{Path(__file__).parent}/golden"

def create_interpreter(model_file, plataform):
    if plataform == Plataform.TensorFlowLite:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()
    return interpreter

def set_interpreter_intput(interpreter, input_image, operation):
    ext = Path(input_image).suffix
    size = common.input_size(interpreter)
    
    if ext == ".npy":
        input = np.load(input_image)

        if len(input.shape) == 2:
            input = input[..., np.newaxis]
    else:
        if operation == Operation.Conv2d:
            imgGray = Image.open(input_image).convert('L').resize(size, Image.ANTIALIAS)
            input = np.array(imgGray).reshape((*size, 1))
        elif operation == Operation.DepthConv2d:
            input = Image.open(input_image).convert('RGB').resize(size, Image.ANTIALIAS)

    common.set_input(interpreter, input)

def get_input_tensor(interpreter):
    input_details = interpreter.get_input_details()
    tensor_in = interpreter.get_tensor(input_details[0]["index"])
    return tensor_in

def get_output_array(interpreter):
    output_details = interpreter.get_output_details()
    out_tensor = interpreter.tensor(output_details[0]["index"])()
    out_array = np.squeeze(out_tensor).astype(np.uint8)
    return out_array

def get_output_image_filename(model_file):
    return f"output_{util.get_model_name(model_file)}.jpg"

def save_output_image(output, model_file):
    out_img_file = get_output_image_filename(model_file)
    Image.fromarray(output).save(out_img_file)
    print(f"Output image saved to `{out_img_file}`")

def get_output_golden_filename(model_file):
    return f"{GOLDEN_DIR}/golden_{util.get_model_name(model_file)}.npy"

def save_output_golden(output, model_file):
    out_gold_file = get_output_golden_filename(model_file)
    np.save(out_gold_file, output)
    print(f"Golden output saved to `{out_gold_file}`")

def check_output_against_golden(output, golden_file):    
    if os.path.isfile(golden_file):
        golden = np.load(golden_file)
        errors = np.sum(output != golden)
        return errors
    else:
        raise FileNotFoundError

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input', required=True,
                        help='Path to input image (.jpg,.png,...)')
    parser.add_argument('-M', '--model', required=True,
                        help='Path to model file (.tflite)')
    parser.add_argument('-G', '--golden', required=False, default=None,
                        help='Path to golden output file (.npy)')
    parser.add_argument('--iterations', required=False, default=10,
                        help='Number of iterations')
    parser.add_argument('--save-golden', default=False, action='store_true',
                        help='Whether the output should be saved to a binary file in .npy format or not')
    parser.add_argument('--save-image', default=False, action='store_true',
                        help='Whether the output should be saved to an image in .jpg format or not')
    args = parser.parse_args()

    model_file = args.model
    input_image_file = args.input
    golden_file = args.golden
    iterations = int(args.iterations)

    dft_golden_file = get_output_golden_filename(model_file)
    plataform = util.get_plataform_from_model_name(model_file)
    operation = util.get_op_from_model_name(model_file)

    # Setup log helper
    benchmarkName = "CoralConv2d"
    benchmarkInfo = f"model_file: {model_file} input_file: {input_image_file} dft_golden_file: {dft_golden_file} iterations: {iterations}"
    lh.start_log_file(benchmarkName, benchmarkInfo)
    lh.set_max_errors_iter(500)
    lh.set_iter_interval_print(1)
    print(f"Log file is: {lh.get_log_file_name()}")

    t0 = time()
    interpreter = create_interpreter(model_file, plataform)

    t1 = time()
    print(f"Create interpreter: {t1 - t0}s")

    for i in range(iterations):
        t1 = time()
        set_interpreter_intput(interpreter, input_image_file, operation)

        t2 = time()
        print(f"Load input: {t2 - t1}s")

        lh.start_iteration()
        interpreter.invoke()
        lh.end_iteration()

        t3 = time()
        print(f"Run interpreter: {t3 - t2}s")

        output = get_output_array(interpreter)

        t4 = time()
        print(f"Get output: {t4 - t3}s")

        if golden_file is None:
            try:
                errors = check_output_against_golden(output, dft_golden_file)
                t5 = time()
                print(f"Check output: {t5 - t4}s - {errors} error(s)")
                lh.log_error_count(int(errors))
            except: pass
        else:
            try:
                errors = check_output_against_golden(output, golden_file)
                t5 = time()
                print(f"Check output: {t5 - t4}s - {errors} error(s)")
                lh.log_error_count(int(errors))
            except FileNotFoundError:
                print(f"Could not open golden file `{golden_file}`")
                lh.log_error_count(f"Could not open golden file `{golden_file}`")

    if args.save_image:
        save_output_image(output, model_file)

    if args.save_golden:
        save_output_golden(output, model_file)
    

if __name__ == "__main__":
    main()