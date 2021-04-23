from enum import Enum
from pathlib import Path
import re

class Plataform(Enum):
    TensorFlowLite = "TFLite"
    EdgeTPU = "EdgeTPU"

class Operation(Enum):
    # Values are the model filename prefix
    Conv2d = "conv_2d"
    DepthConv2d = "depthwise_conv_2d"

    def flatbuffers_code(self):
        return self.value.upper()

def get_model_name(model_file):
    return Path(model_file).stem

def get_dims_from_model_name(model_filename):
    return list(map(int, re.findall("(\d+)[_\.]", model_filename)))

def get_plataform_from_model_name(model_filename):
    return Plataform.EdgeTPU if model_filename.endswith("_edgetpu.tflite") else Plataform.TensorFlowLite

def get_op_from_model_name(model_filename):
    operation = None
    for op in Operation:
        if Path(model_filename).stem.startswith(op.value):
            operation = op
            break
    assert operation is not None, "Unsupported operation"
    return operation

def get_root_path():
    return Path(__file__).absolute().parent.parent

def get_inputs_path():
    return Path(f"{get_root_path()}/inputs")

def get_models_path():
    return Path(f"{get_root_path()}/models")

def get_scripts_path():
    return Path(f"{get_root_path()}/scripts")