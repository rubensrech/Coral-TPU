import sys
import argparse
from enum import Enum

from pycoral.adapters import common

import numpy as np
from PIL import Image

def save_image(tensor, path):
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    Image.fromarray(np.squeeze(tensor).astype(np.uint8), "RGB").save(path)

class Plataform(Enum):
    TensorFlowLite = "TFLite"
    EdgeTPU = "EdgeTPU"

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-M', '--model', required=True,
                        help='Path to model file (.tflite)')
    parser.add_argument('-I', '--input', required=True,
                        help='Path to input image (.jpg,.png,...)')
    args = parser.parse_args()

    model_file = args.model
    input_image = args.input
    plataform = Plataform.EdgeTPU if model_file.endswith("_edgetpu.tflite") else Plataform.TensorFlowLite

    # Create interpreter
    if plataform == Plataform.TensorFlowLite:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)

    # Set input to the interpreter
    input = Image.open(input_image).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, input)

    # Get raw input tensor
    # input_details = interpreter.get_input_details()
    # tensor_in = interpreter.get_tensor(input_details[0]["index"])

    # Run interpreter
    interpreter.invoke()

    # Get output
    output_details = interpreter.get_output_details()
    tensor_out = interpreter.get_tensor(output_details[0]["index"])
    output = tensor_out.copy()
    
    # Save output image
    out_file = f"depthwise_conv2d_result_{plataform.value}.jpg"
    save_image(output, out_file)
    print(f"Result saved to `{out_file}`")


if __name__ == "__main__":
    main()