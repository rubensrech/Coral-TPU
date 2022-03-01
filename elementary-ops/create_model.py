#!/usr/bin/env python3

import os
import json
import logging
import argparse
from re import T
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf

from src.util import MODELS_DIR, SCRIPTS_DIR
from src.util import Operation, Plataform, OperationOptions, Kernel
from src.util import echo_run, get_model_filename, get_model_name, get_path_relative_to_install_dir, avg_kernel, ones_kernel

REL_MODELS_DIR = get_path_relative_to_install_dir(MODELS_DIR)
REL_SCRIPTS_DIR = get_path_relative_to_install_dir(SCRIPTS_DIR)

np.random.seed(0)

log = logging.getLogger("OpModelCreator")
log.setLevel(logging.INFO)

# TensorFlow model

def create_depthwise_conv2d_tf(kernel: tf.Tensor, input_shape: Tuple[int]):
    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
    def depthwise_conv2d(input):
        return tf.nn.depthwise_conv2d(input, kernel, strides=[1] * 4, padding="SAME")
    
    return depthwise_conv2d

def create_conv2d_tf(kernel: tf.Tensor, input_shape: Tuple[int]):
    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
    def conv2d(input):
        return tf.nn.conv2d(input, kernel, strides=[1] * 4, padding="SAME")
    
    return conv2d

def create_add_tf(input_shape: Tuple[int]):
    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
    def add(A):
        B = np.ones(input_shape, dtype=np.float32)
        return tf.add(A, B)
    
    return add

# TensorFlow Lite model

def create_tflite_model(model_name: str, model_func):
    model_file = get_model_filename(model_name)
    log.info(f"Generating the TensorFlow Lite model ({model_file})")
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([model_func.get_concrete_function()])
    tflite_model = converter.convert()

    with open(model_file, "wb") as fout:
        fout.write(tflite_model)

    return model_file

def create_op_tflite_model(op: Operation, kernel: tf.Tensor, input_shape: Tuple[int]):
    model_name = get_model_name(op, input_shape, kernel.shape, Plataform.TensorFlowLite)

    # Create TF Lite model
    if op == Operation.DepthConv2d:
        model_creator = lambda: create_depthwise_conv2d_tf(kernel, input_shape)
    elif op == Operation.Conv2d:
        model_creator = lambda: create_conv2d_tf(kernel, input_shape)

    model_func = model_creator()
    model_file = create_tflite_model(model_name, model_func)

    return model_file

# Edge TPU model

def create_edgetpu_model(input_shape: Tuple[int], model_name: str, model_func, keep_only_op_codes: List[str]):
    def gen_input_samples():
        yield [np.zeros(input_shape, np.float32)]
        yield [np.random.random_sample(input_shape).astype(np.float32) * 255]
        yield [np.ones(input_shape, np.float32) * 255]

    log.info("Generating the quantized TensorFlow Lite model")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([model_func.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = gen_input_samples
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    quant_model_name = model_name.rstrip('edgetpu').rstrip('_')
    quant_model_file = get_model_filename(quant_model_name, relative_to_install_dir=True)
    with open(quant_model_file, "wb") as fout:
        fout.write(tflite_model)
    log.info("Wrote quantized TensorFlow Lite model to %s", quant_model_file)

    # Patching the standard TensorFlow Lite model
    if not Path("schema.fbs").exists():
        log.info("`schema.fbs` was not found, downloading")
        urllib.request.urlretrieve(
            "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
            "schema.fbs")
        log.info("Downloaded `schema.fbs`")

    log.info("Converting the model from binary flatbuffers to JSON")
    echo_run("flatc", "-t", "--strict-json", "--defaults-json", "-o", REL_MODELS_DIR, "schema.fbs", "--", quant_model_file)

    log.info("Patching the model in JSON")
    quant_model_file_json = str(Path(quant_model_file).with_suffix(".json"))
    with open(quant_model_file_json) as fin:
        model = json.load(fin)

    # Erase all opcodes except the ones listed in `keep_only_op_codes`
    if len(keep_only_op_codes) > 0:
        conv_opcode = -1
        new_opcodes = []
        for i, c in enumerate(model["operator_codes"]):
            if c["builtin_code"] in keep_only_op_codes:
                new_opcodes.append(c)
                conv_opcode = i
        assert conv_opcode >= 0
        model["operator_codes"] = new_opcodes
    
    # Fix the tensor dtypes which are int8 instead of uint8
    # Also remove the multi-channel quantization which is not supported on Edge TPU
    graph = model["subgraphs"][0]
    new_tensors = []
    index_map = {}
    for i, t in enumerate(graph["tensors"]):
        if t["type"] == "FLOAT32":
            continue
        if t["type"] == "INT8":
            t["type"] = "UINT8"
            t["quantization"]["zero_point"][0] = 0
        t["quantization"]["scale"] = [t["quantization"]["scale"][0]]
        t["quantization"]["zero_point"] = [t["quantization"]["zero_point"][0]]
        t["quantization"]["quantized_dimension"] = 0
        index_map[i] = len(new_tensors)
        new_tensors.append(t)
    graph["tensors"] = new_tensors

    # Update the tensor indexes in rhe ops
    new_ops = []
    for op in graph["operators"]:
        if op["opcode_index"] != conv_opcode:
            continue
        op["outputs"] = [index_map[i] for i in op["outputs"]]
        op["inputs"] = [index_map[i] for i in op["inputs"]]
        new_ops.append(op)
    graph["operators"] = new_ops

    # Update the global input and output tensor indexes
    graph["inputs"][0] = new_ops[0]["inputs"][0]
    graph["outputs"][0] = new_ops[0]["outputs"][0]
    model["subgraphs"][0] = graph

    # Write patched JSON model to file
    with open(quant_model_file_json, "w") as fout:
        json.dump(model, fout, indent=4)

    log.info("Generating the binary flatbuffers model from JSON")
    echo_run("flatc", "-b", "-o", REL_MODELS_DIR, "schema.fbs", quant_model_file_json)

    log.info("Compiling the Edge TPU model")
    echo_run(f"{REL_SCRIPTS_DIR}/edgetpu_compiler.sh", "-s", "-o", REL_MODELS_DIR, quant_model_file)
    Path(quant_model_file_json).unlink()
    Path(quant_model_file).with_name(Path(quant_model_file).stem + "_edgetpu.log").unlink()

    return get_model_filename(model_name)

def create_op_edgetpu_model(op: Operation, input_shape: Tuple[int], op_params: Dict):
    model_name = None
    model_creator = None
    op_code = op.flatbuffers_code()

    if op in [Operation.Conv2d, Operation.DepthConv2d]:
        if not op_params['kernel']:
            raise RuntimeError(f"Parameter `kernel` is required for Operation [{op}]")
        
        kernel = op_params['kernel']
        model_name = get_model_name(op, input_shape, Plataform.EdgeTPU, op_attrs={ 'kernel_shape': kernel.shape })

        if op == Operation.DepthConv2d:
            model_creator = lambda: create_depthwise_conv2d_tf(kernel, input_shape)
        elif op == Operation.Conv2d:
            model_creator = lambda: create_conv2d_tf(kernel, input_shape)
    elif op == Operation.Add:
        model_name = get_model_name(op, input_shape, Plataform.EdgeTPU)
        model_creator = lambda: create_add_tf(input_shape)

    if not model_name or not model_creator:
        raise RuntimeError(f"Unrecognized Operation [{op}]")

    model_func = model_creator()
    model_file = create_edgetpu_model(input_shape, model_name, model_func, keep_only_op_codes=[op_code])   

    return model_file

def get_input_shape(input_size: Tuple[int], op: Operation):
    assert len(input_size) == 2, "Input size must be rank 2"

    if op == Operation.DepthConv2d:
        input_shape = (1, *input_size, 3)   # 3 channels (R,G,B)
    elif op == Operation.Conv2d:        
        input_shape = (1, *input_size, 1)   # 1 channel
    elif op == Operation.Add:
        input_shape = input_size            # 2 dims
    else:
        raise RuntimeError(f"Unrecognized Operation [{op}]")
    
    return input_shape

def create_kernel(kernel_type: Kernel, kernel_size: Tuple[int], op: Operation):
    assert len(kernel_size) == 2, "Kernel size must be rank 2"

    if op == Operation.DepthConv2d:
        if kernel_type == Kernel.Average:    kernel = avg_kernel(kernel_size)
        elif kernel_type == Kernel.Ones:     kernel = ones_kernel(kernel_size)
        kernel = tf.tile(tf.constant(kernel)[:, :, None, None], [1, 1, 3, 1]) # 3 channels (R,G,B)

    elif op == Operation.Conv2d:
        # Tensorflow needs `kernel_shape` to be rank 4
        kernel_shape = (*kernel_size, 1, 1)

        if kernel_type == Kernel.Average:    kernel = avg_kernel(kernel_shape)
        elif kernel_type == Kernel.Ones:     kernel = ones_kernel(kernel_shape)

    return kernel

def create_op_model(op: Operation, input_size: Tuple[int], op_options: OperationOptions, plataform: Plataform):
    op_params = {}

    # Tensorflow needs `input_shape` to be rank 4
    input_shape = get_input_shape(input_size, op)

    if op in [Operation.Conv2d, Operation.DepthConv2d]:
        op_params['kernel'] = create_kernel(op_options.kernel_type, op_options.kernel_size, op)

    # Create MODELS_DIR, if it does not exist
    if not os.path.isdir(MODELS_DIR): echo_run("mkdir", "-p", MODELS_DIR)

    # Create model
    if plataform == Plataform.TensorFlowLite:
        return create_op_tflite_model(op, op_params['kernel'], input_shape) ##################################
    elif plataform == Plataform.EdgeTPU:
        return create_op_edgetpu_model(op, input_shape, op_params)

def validate_args(args):
    def error_and_exit(msg: str):
        print(msg)
        exit(-1)

    # Required arguments

    # - Operation
    op = Operation(args.operation.lower())

    # - Input Size
    input_dims = args.input_size.split(",")
    if len(input_dims) != 2:
        error_and_exit(f"INPUT SIZE must have exactly 2 dimensions (H,W)")
    input_size = tuple(map(int, args.input_size.split(",")))

    # Operation dependent arguments

    op_opts = OperationOptions()

    if op in [Operation.DepthConv2d, Operation.Conv2d]:
        # - Kernel Size
        if not args.kernel_size:
            error_and_exit(args.kernel_size, f"KERNEL SIZE is required for OPERATION [{op}]")
        op_opts.kernel_size = tuple(map(int, args.kernel_size.split(",")))

        # - Kernel Type
        if not args.kernel_name:
            error_and_exit(args.kernel_name, f"KERNEL NAME is required for OPERATION [{op}]")
        op_opts.kernel_type = Kernel(args.kernel_name)

    return op, input_size, op_opts

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-O', '--operation', required=True,
                        help='Operation: DEPTHWISE_CONV_2D | CONV_2D | ADD')
    parser.add_argument('-I', '--input-size', required=True,
                        help='Input size (format: H,W)')
    parser.add_argument('-K', '--kernel-size', required=False,
                        help='Kernel size (format: H,W)')
    parser.add_argument('-N', '--kernel-name', default="AVERAGE", required=False,
                        help='Kernel name: AVERAGE | ONES')
    parser.add_argument('-P', '--plataform', default="BOTH",
                        help='Plataform: TFLite | EdgeTPU | BOTH')
    args = parser.parse_args()

    op, input_size, op_opts = validate_args(args)


    print("")
    print(f"Operation: {op}")
    print(f"Input size: {input_size}")
    print(f"Operation options: {op_opts}")

    def create_model_for_plataform(plataform: Plataform):
        model_file = create_op_model(op, input_size, op_opts, plataform)
        print(f"Plataform: {plataform}")
        print(f"Model successfully saved to `{model_file}`")
    
    if args.plataform == "BOTH":
        for plataform in Plataform:
            print("")
            create_model_for_plataform(plataform)
    else:
        plataform = Plataform(args.plataform)
        create_model_for_plataform(plataform)

if __name__ == "__main__":
    main()