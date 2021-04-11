import os
import json
import logging
import argparse
import subprocess
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import tensorflow as tf

OUT_DIR = f"{Path(__file__).parent}/models/"

log = logging.getLogger("OpModelCreator")
log.setLevel(logging.INFO)

# Execute an arbitrary command and echo its output
def echo_run(*cmd):
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output:
        print(output)
    p.check_returncode()

# Kernels

def ones_kernel(size: Tuple[int]) -> tf.Tensor:
    return np.ones(size, dtype=np.float32)


def avg_kernel(size: Tuple[int]) -> tf.Tensor:
    (H, W, *c) = size
    return np.ones((H, W), dtype=np.float32)/(H*W)

# TensorFlow model

def create_depthwise_conv2d_tf(kernel: tf.Tensor, input_shape: Tuple[int]):
    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
    def depthwise_conv2d(input):
        return tf.nn.depthwise_conv2d(input, kernel, strides=[1] * 4, padding="SAME")
    
    return depthwise_conv2d

# TensorFlow Lite model

def create_tflite_model(input_shape: Tuple[int], model_name: str, model_func):
    model_file = f"{OUT_DIR}{model_name}.tflite"
    log.info(f"Generating the TensorFlow Lite model ({model_file})")
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([model_func.get_concrete_function()])
    tflite_model = converter.convert()

    with open(model_file, "wb") as fout:
        fout.write(tflite_model)

    return model_file

def create_depthwise_conv2d_tflite_model(kernel: tf.Tensor, input_shape: Tuple[int]):
    shape_str = "_".join(map(str, input_shape))
    model_name = "depthwise_conv2d_%s" % (shape_str)

    # Create TF Lite model
    model_creator = lambda: create_depthwise_conv2d_tf(kernel, input_shape)
    model_func = model_creator()
    model_file = create_tflite_model(input_shape, model_name, model_func)

    return model_file

# Edge TPU model

def create_edgetpu_model(input_shape: Tuple[int], model_name: str, model_func, keep_only_op_codes=["DEPTHWISE_CONV_2D"]):
    def gen_input_samples():
        yield [np.zeros(input_shape, np.float32)]
        yield [np.ones(input_shape, np.float32) * 255]

    log.info("Generating the quantized TensorFlow Lite model ")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([model_func.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = gen_input_samples
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    quant_model_file = f"{OUT_DIR}{model_name}.tflite"
    with open(quant_model_file, "wb") as fout:
        fout.write(tflite_model)
    log.info("Wrote quantized TensorFlow Lite model to %s", quant_model_file)

    # Patching the standard TensorFlow Lite model
    if not Path("schema.fbs").exists():
        log.info("schema.fbs was not found, downloading")
        urllib.request.urlretrieve(
            "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
            "schema.fbs")
        log.info("Downloaded schema.fbs")

    log.info("Converting the model from binary flatbuffers to JSON")
    echo_run("flatc", "-t", "--strict-json", "--defaults-json", "-o", OUT_DIR, "schema.fbs", "--", quant_model_file)

    log.info("Patching the model in JSON")
    quant_model_file_json = str(Path(quant_model_file).with_suffix(".json"))
    with open(quant_model_file_json) as fin:
        model = json.load(fin)

    # Erase all opcodes except the ones listed in `keep_only_op_codes`
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

    with open(quant_model_file_json, "w") as fout:
        json.dump(model, fout, indent=4)
    
    log.info("Generating the binary flatbuffers model from JSON")
    echo_run("flatc", "-b", "-o", OUT_DIR, "schema.fbs", quant_model_file_json)

    log.info("Compiling the Edge TPU model")
    echo_run("edgetpu_compiler", "-s", "-o", OUT_DIR, quant_model_file)
    Path(quant_model_file_json).unlink()
    Path(quant_model_file).with_name(Path(quant_model_file).stem + "_edgetpu.log").unlink()

    model_file = Path(quant_model_file).with_name(Path(quant_model_file).stem + "_edgetpu.tflite")
    return model_file

def create_depthwise_conv2d_edgetpu_model(kernel: tf.Tensor, input_shape: Tuple[int]):
    shape_str = "_".join(map(str, input_shape))
    model_name = "depthwise_conv2d_%s_quant" % (shape_str)

    # Create Edge TPU model
    model_creator = lambda: create_depthwise_conv2d_tf(kernel, input_shape)
    model_func = model_creator()
    model_file = create_edgetpu_model(input_shape, model_name, model_func)   

    return model_file

class Plataform(Enum):
    TensorFlowLite = "TFLite"
    EdgeTPU = "EdgeTPU"

class Kernel(Enum):
    Average = "AVERAGE"
    Ones = "ONES"

def create_depthwise_conv2d_model(input_size: Tuple[int], kernel_size: Tuple[int], kernel_type: Kernel, plataform: Plataform):
    # Tensorflow needs `input_shape` to be rank 4
    assert len(input_size) == 3, "Input size must be rank 3"
    input_shape = (1, *input_size)
    C = input_shape[3]

    assert len(kernel_size) == 2, "Kernel shape must be rank 2"

    # Create kernel
    kernel = None
    if kernel_type == Kernel.Average: kernel = avg_kernel(kernel_size)
    assert kernel is not None, f"Unknown kernel `{kernel_type}`"

    # Adjust kernel dims
    kernel = tf.tile(tf.constant(kernel)[:, :, None, None], [1, 1, C, 1])

    # Create OUT_DIR, if it does not exist
    if not os.path.isdir(OUT_DIR): echo_run("mkdir", "-p", OUT_DIR)

    # Create model
    if plataform == Plataform.TensorFlowLite:
        return create_depthwise_conv2d_tflite_model(kernel, input_shape)
    elif plataform == Plataform.EdgeTPU:
        return create_depthwise_conv2d_edgetpu_model(kernel, input_shape)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input-size', default="300,300,3",
                        help='Input size (format: H,W,C)')
    parser.add_argument('-K', '--kernel-size', default="15,15",
                        help='Kernel size (format: H,W)')
    parser.add_argument('-N', '--kernel-name', default="AVERAGE",
                        help='Kernel name: AVERAGE | ONES')
    parser.add_argument('-P', '--plataform', default="BOTH",
                        help='Plataform: TFLite | EdgeTPU | BOTH')
    args = parser.parse_args()

    input_size = tuple(map(int, args.input_size.split(",")))
    kernel_size = tuple(map(int, args.kernel_size.split(",")))
    kernel_type = Kernel(args.kernel_name)

    print("")
    print(f"Input size: {input_size}")
    print(f"Kernel size: {kernel_size}")
    print(f"Kernel type: {kernel_type}")

    def create_model_for_plataform(plataform: Plataform):
        model_file = create_depthwise_conv2d_model(input_size, kernel_size, kernel_type, plataform)
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