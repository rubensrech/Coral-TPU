import sys
import argparse
import numpy as np
from enum import Enum
from time import time
from PIL import Image
from pathlib import Path
from pycoral.adapters import common


def save_image(tensor, path):
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    Image.fromarray(np.squeeze(tensor).astype(np.uint8)).save(path)

class Plataform(Enum):
    TensorFlowLite = "TFLite"
    EdgeTPU = "EdgeTPU"

class Operation(Enum):
    # Values are the model filename prefix
    Conv2d = "conv_2d"
    DepthConv2d = "depthwise_conv_2d"

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

    operation = None
    for op in Operation:
        if Path(model_file).stem.startswith(op.value):
            operation = op
            break
    
    assert operation is not None, "Unsupported operation"

    t0 = time()

    # Create interpreter
    if plataform == Plataform.TensorFlowLite:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)

    t1 = time()

    # Set input to the interpreter
    if operation == Operation.Conv2d:
        imgGray = Image.open(input_image).convert('LA').resize(size, Image.ANTIALIAS)
        input = np.array(imgGray)[:,:,0].reshape((*size, 1))
    elif operation == Operation.DepthConv2d:
        input = Image.open(input_image).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, input)

    t2 = time()

    # Get raw input tensor
    # input_details = interpreter.get_input_details()
    # tensor_in = interpreter.get_tensor(input_details[0]["index"])

    # Run interpreter
    interpreter.invoke()

    t3 = time()

    # Get output
    output_details = interpreter.get_output_details()
    tensor_out = interpreter.get_tensor(output_details[0]["index"])
    output = tensor_out.copy()
    
    # Save output image
    out_file = f"{operation.value}_{plataform.value}_output.jpg"
    save_image(output, out_file)

    t4 = time()

    print(f"Load interpreter: {t1 - t0} s")
    print(f"Load input: {t2 - t1} s")
    print(f"Run interpreter: {t3 - t2} s")
    print(f"Save output: {t4 - t3} s")
    print(f"Output saved to `{out_file}`")


if __name__ == "__main__":
    main()