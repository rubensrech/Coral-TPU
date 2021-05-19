#!/usr/bin/env python3

import os
import re
import json
import subprocess
from pathlib import Path

JSON_FILE = "/home/carol/radiation-benchmarks/scripts/json_files/coralElementaryOps.json"
JSON_PARAM = "/home/carol/radiation-benchmarks/scripts/json_files/json_parameter"

INSTALL_DIR = Path(__file__).parent.absolute()
MODELS_DIR = f"{INSTALL_DIR}/models"
INPUTS_DIR = f"{INSTALL_DIR}/inputs"

def echo_run(args_list):
    p = subprocess.run(args_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output: print(output)
    return output

def get_full_path(filename):
    return Path(filename).absolute()

BENCHMARKS_JSON_LIST = []
BENCHMARKS_MODELS = []

model_files = os.listdir(MODELS_DIR)
model_files = [
    'conv_2d_1_256_256_1_40_40_1_1_quant_edgetpu.tflite',
    'conv_2d_1_512_512_1_40_40_1_1_quant_edgetpu.tflite',
    'conv_2d_1_1024_1024_1_40_40_1_1_quant_edgetpu.tflite',
    'depthwise_conv_2d_1_256_256_1_20_20_1_1_quant_edgetpu.tflite',
    'depthwise_conv_2d_1_512_512_1_20_20_1_1_quant_edgetpu.tflite',
    'depthwise_conv_2d_1_1024_1024_1_20_20_1_1_quant_edgetpu.tflite',
]

for model_filename in model_files:
    if model_filename.endswith("edgetpu.tflite"):
        full_path_to_model = f"{MODELS_DIR}/{model_filename}"

        # Generate input
        print(f"GENERATING INPUT FOR MODEL `{model_filename}`")
        gen_in_out = echo_run(['python3', 'prepare_input_for_model.py', '-M', full_path_to_model])
        path_to_input = re.compile('`(.*)`').findall(gen_in_out)[0]
        full_path_to_input = get_full_path(path_to_input)

        # Generate golden
        print(f"GENERATING GOLDEN FOR MODEL `{model_filename}`")
        gen_gold_out = echo_run(['./run_model', full_path_to_model, full_path_to_input, '--save-golden', '1'])
        path_to_gold = re.compile('INFO: Golden output saved to `(.*)`').findall(gen_gold_out)[0]
        full_path_to_gold = get_full_path(path_to_gold)

        full_path_to_bin = f"{INSTALL_DIR}/run_model"

        benchmark_exec_cmd = f"sudo {full_path_to_bin} {full_path_to_model} {full_path_to_input} {full_path_to_gold} --iterations 1000000000"
        benchmark_kill_cmd = "killall -9 run_model"
        BENCHMARKS_JSON_LIST.append({ "exec": benchmark_exec_cmd, "killcmd": benchmark_kill_cmd })
        BENCHMARKS_MODELS.append(model_filename)

with open(JSON_FILE, 'w') as outfile:
    json.dump(BENCHMARKS_JSON_LIST, outfile, indent=4)

with open(JSON_PARAM, 'w') as f:
    f.write(JSON_FILE)

print(f"{len(BENCHMARKS_MODELS)} BENCHMARKS GENERATED:")
print(json.dumps(BENCHMARKS_MODELS, indent=4))
print(f"BENCHMARKS JSON PARAMETER WRITTEN TO `{JSON_FILE}`")
print(f"JSON PARAMETER UPDATED")