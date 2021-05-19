#!/usr/bin/env python3

import os
import re
import json
import subprocess
from pathlib import Path

JSON_FILE = "/home/carol/radiation-benchmarks/scripts/json_files/detection.json"
JSON_PARAM = "/home/carol/radiation-benchmarks/scripts/json_files/json_parameter"

INSTALL_DIR = Path(__file__).parent.absolute()
MODELS_DIR = f"{INSTALL_DIR}/models"
INPUTS_DIR = f"{INSTALL_DIR}/inputs"

def echo_run(args):
    if type(args) == str:
        args = args.split(' ')
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output: print(output)
    return output

def get_full_path(filename):
    return Path(filename).absolute()


BENCHMARKS_JSON_LIST = []
BENCHMARKS_DESCRIPTORS = [
    { 'model': 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite', 'inputs-dir': 'inputs/test-imgs' },
    # { 'model': 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', 'inputs-dir': '/home/carol/radiation-benchmarks/data/VOC2012', 'nimages': 100  }
]

for benchmark in BENCHMARKS_DESCRIPTORS:
    # Create input images list
    nimages = benchmark.get('nimages', None)
    inputs_dir = benchmark['inputs-dir']
    print(f"GENERATING INPUT IMAGES LIST FROM `{inputs_dir}`")
    if nimages is not None:
        gen_in_out = echo_run(f"python3 create_inputs_list.py {inputs_dir} -n {nimages}")
    else:
        gen_in_out = echo_run(f"python3 create_inputs_list.py {inputs_dir}")
    path_to_input = re.compile('`(.*)`').findall(gen_in_out)[0]
    full_path_to_input = get_full_path(path_to_input)

    # Generate golden
    model_filename = benchmark['model']
    full_path_to_model = MODELS_DIR + '/' + model_filename
    print(f"GENERATING GOLDEN FOR MODEL `{model_filename}`")
    echo_run(f"python3 run_detection.py --model {full_path_to_model} --input {full_path_to_input} --save-golden")

    full_path_to_script = get_full_path("run_detection.py")
    benchmark_exec_cmd = f"sudo python3 {full_path_to_script} --model {full_path_to_model} --input {full_path_to_input} --iterations 1000000000"
    benchmark_kill_cmd = "pkill -9 -f run_detection.py"
    BENCHMARKS_JSON_LIST.append({ "exec": benchmark_exec_cmd, "killcmd": benchmark_kill_cmd })

with open(JSON_FILE, 'w') as outfile:
    json.dump(BENCHMARKS_JSON_LIST, outfile, indent=4)

with open(JSON_PARAM, 'w') as f:
    f.write(JSON_FILE)

print(f"{len(BENCHMARKS_JSON_LIST)} BENCHMARKS GENERATED:")
print(json.dumps(BENCHMARKS_DESCRIPTORS, indent=4))
print(f"BENCHMARKS JSON PARAMETER WRITTEN TO `{JSON_FILE}`")
print(f"JSON PARAMETER UPDATED")