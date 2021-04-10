import sys
from enum import Enum

from pycoral.adapters import common

import numpy as np
from PIL import Image

def save_image(tensor, path):
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    Image.fromarray(np.squeeze(tensor).astype(np.uint8), "RGB").save(path)

class Plataform(Enum):
    TensorFlowLite = 1
    EdgeTPU = 2

def main():
    model_file = sys.argv[1]
    input_image = "../couscous.jpg"
    plataform = Plataform.EdgeTPU

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
    save_image(output, "depthwise_conv2d_result.jpg")


if __name__ == "__main__":
    main()