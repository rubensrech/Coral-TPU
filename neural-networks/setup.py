#!/usr/bin/env python3

import re
import json
import subprocess
from pathlib import Path

# Better run with `sudo`:
# > sudo ./setup.py

################################################################
# > Configuration
# Add or comment benchmarks to be generated in the
# correspondent task in the dictionary below

BENCHMARKS_DESCRIPTORS = {
    'detection': [
        { 'model': 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite', 'inputs-dir': '/home/carol/radiation-benchmarks/data/VOC2012', 'nimages': 100 },
        { 'model': 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', 'inputs-dir': '/home/carol/radiation-benchmarks/data/VOC2012', 'nimages': 100 },
        { 'model': 'ssd_mobilenet_v2_catsdogs_quant_edgetpu.tflite', 'inputs-dir': '/home/carol/oxford-pets-100' },
        { 'model': 'ssd_mobilenet_v2_transf_learn_catsdogs_quant_edgetpu.tflite', 'inputs-dir': '/home/carol/oxford-pets-100' },
        { 'model': 'ssd_mobilenet_v2_subcoco14_quant_edgetpu.tflite', 'inputs-dir': '/home/carol/subcoco14' },
        { 'model': 'ssd_mobilenet_v2_subcoco14_transf_learn_quant_edgetpu.tflite', 'inputs-dir': '/home/carol/subcoco14' },
    ],
    'classification': [
        { 'model': 'tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite', 'inputs-dir': '/home/carol/ILSVRC2012_val_100' },
        { 'model': 'inception_v4_299_quant_edgetpu.tflite', 'inputs-dir': '/home/carol/ILSVRC2012_val_100' }
    ],
}

JSON_FILES_PATH = "/home/carol/radiation-benchmarks/scripts/json_files"
JSON_PARAM = f"{JSON_FILES_PATH}/json_parameter"

################################################################

# Paths

INSTALL_DIR = Path(__file__).parent.absolute()
MODELS_DIR = f"{INSTALL_DIR}/models"
INPUTS_DIR = f"{INSTALL_DIR}/inputs"

# Util functions

def echo_run(args):
    if type(args) == str:
        args = args.split(' ')
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output: print(output)
    return output

def get_full_path(filename):
    return Path(filename).absolute()

def get_script_for_task(task):
    return "run_" + task + ".py"

def get_path_to_json_file_for_task(task):
    return f"{JSON_FILES_PATH}/{task}.json"

# Main functions

def setup_benchmarks(benchmark_descriptors):
    generated_benchmarks_grouped_by_task = {}

    for task in benchmark_descriptors:
        generated_benchmarks_grouped_by_task[task] = {
            'benchmarks_data': [],
            'json_file': get_path_to_json_file_for_task(task)
        }

        task_script = get_script_for_task(task)

        for benchmark in benchmark_descriptors[task]:
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
            echo_run(f"sudo python3 {task_script} --model {full_path_to_model} --input {full_path_to_input} --save-golden")

            # Build JSON data
            full_path_to_script = get_full_path(task_script)
            benchmark_exec_cmd = f"sudo python3 {full_path_to_script} --model {full_path_to_model} --input {full_path_to_input} --iterations 1000000000"
            benchmark_kill_cmd = f"pkill -9 -f {task_script}"
            benchmark_json_data = { "exec": benchmark_exec_cmd, "killcmd": benchmark_kill_cmd }

            generated_benchmarks_grouped_by_task[task]['benchmarks_data'].append(benchmark_json_data)

        # Write benchmarks to JSON file
        task_json_file = generated_benchmarks_grouped_by_task[task]['json_file']
        task_benchmarks_data = generated_benchmarks_grouped_by_task[task]['benchmarks_data']

        with open(task_json_file, 'w') as f:
            json.dump(task_benchmarks_data, f, indent=4)

    return generated_benchmarks_grouped_by_task

def update_json_param(generated_benchmarks_grouped_by_task):
    generated_json_files = list(map(lambda t: generated_benchmarks_grouped_by_task[t]['json_file'], generated_benchmarks_grouped_by_task))

    with open(JSON_PARAM, 'w') as f:
        for json_file in generated_json_files:
            f.write(json_file + '\n')

def main():
    generated_benchmarks_grouped_by_task = setup_benchmarks(BENCHMARKS_DESCRIPTORS)
    update_json_param(generated_benchmarks_grouped_by_task)

    print("--------------------- SUMMARY ---------------------")
    for task in generated_benchmarks_grouped_by_task:
        nbenchmarks = len(generated_benchmarks_grouped_by_task[task]['benchmarks_data'])
        print(f"{task.upper()}: {nbenchmarks} BENCHMARKS GENERATED")
        print(json.dumps(BENCHMARKS_DESCRIPTORS[task], indent=4))

    print(f"JSON PARAMETER UPDATED ({JSON_PARAM})")

if __name__ == "__main__":
    main()
