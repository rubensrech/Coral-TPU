# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to work with any model."""

import re
import time
import numpy as np
from pathlib import Path

from PIL import Image

from src.utils import detection


def output_tensor(interpreter, i):
    """Gets a model's ith output tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      i (int): The index position of an output tensor.
    Returns:
      The output tensor at the specified position.
    """
    return interpreter.tensor(interpreter.get_output_details()[i]['index'])()

def input_details(interpreter, key):
    """Gets a model's input details by specified key.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      key (int): The index position of an input tensor.
    Returns:
      The input details.
    """
    return interpreter.get_input_details()[0][key]


def input_size(interpreter):
    """Gets a model's input size as (width, height) tuple.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor size as (width, height) tuple.
    """
    _, height, width, _ = input_details(interpreter, 'shape')
    return width, height


def input_tensor(interpreter):
    """Gets a model's input tensor view as numpy array of shape (height, width, 3).
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor view as :obj:`numpy.array` (height, width, 3).
    """
    tensor_index = input_details(interpreter, 'index')
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      data: The input tensor.
    """
    input_tensor(interpreter)[:, :] = data


def set_resized_input(interpreter, resized_image):
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    w, h = resized_image.size
    tensor[:h, :w] = np.reshape(resized_image, (h, w, channel))

def resize_input(image, interpreter):
    width, height = input_size(interpreter)
    w, h = image.size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    resized = image.resize((w, h), Image.ANTIALIAS)
    return resized, (scale, scale)

def read_label_file(file_path):
    """Reads labels from a text file and returns it as a dictionary.
    This function supports label files with the following formats:
    + Each line contains id and description separated by colon or space.
        Example: ``0:cat`` or ``0 cat``.
    + Each line contains a description only. The returned label id's are based on
        the row number.
    Args:
        file_path (str): path to the label file.
    Returns:
        Dict of (int, string) which maps label id to description.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else:
            ret[row_number] = pair[0].strip()
    return ret


###############

INSTALL_DIR = Path(__file__).parent.parent.parent.absolute()
GOLDEN_DIR = f"{INSTALL_DIR}/golden"
INPUTS_DIR = f"{INSTALL_DIR}/inputs"
MODELS_DIR = f"{INSTALL_DIR}/models"
LABELS_DIR = f"{INSTALL_DIR}/labels"
OUTPUTS_DIR = f"{INSTALL_DIR}/outputs"

def get_model_file_from_name(model_name):
    return f"{MODELS_DIR}/{model_name}.tflite"

def get_input_file_from_name(img_name, ext="jpg"):
    return f"{INPUTS_DIR}/{img_name}.{ext}"

def get_dft_golden_filename(model_file: str, image_file: str, ext="npy") -> str:
    model_name = Path(model_file).stem
    image_name = Path(image_file).stem
    return f"{GOLDEN_DIR}/{model_name}--{image_name}.{ext}"

def parse_golden_filename(golden_file: str) -> tuple:
    golden_name = Path(golden_file).stem
    parts = golden_name.split("--")
    model_name = parts[0]
    image_name = parts[1]
    return model_name, image_name

def get_dft_sdc_out_filename(model_file: str, image_file: str, ext="npy") -> str:
    model_name = Path(model_file).stem
    image_name = Path(image_file).stem
    timestap_ms = int(time.time() * 1000)
    return f"{OUTPUTS_DIR}/sdc--{model_name}--{image_name}--{timestap_ms}.{ext}"

def parse_sdc_out_filename(sdc_file: str) -> tuple:
    sdc_name = Path(sdc_file).stem
    parts = sdc_name.split("--")
    if parts[0] == "sdc":
        model_name = parts[1]
        image_name = parts[2]
        timestamp_ms = int(parts[3])
        return model_name, image_name, timestamp_ms
    else:
        raise Exception(f"Invalid SDC file `{sdc_file}`")

def save_tensors_to_file(tensors_dict: dict, filename: str):
    np.save(filename, tensors_dict)

def load_tensors_from_file(filename: str) -> dict:
    return np.load(filename, allow_pickle=True).item()


def get_raw_output(interpreter, coral_out_tensors_idxs=[]) -> dict:
    det_out = detection.get_detection_raw_output(interpreter)._asdict()
    coral_out = { tensorIdx: interpreter.tensor(tensorIdx)() for tensorIdx in coral_out_tensors_idxs }
    return { **det_out, **coral_out }


def create_interpreter(model_file, cpu=False):
    if cpu:
        from tensorflow import lite as tflite
        interpreter = tflite.Interpreter(model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter        
        interpreter = make_interpreter(model_file)

    return interpreter
