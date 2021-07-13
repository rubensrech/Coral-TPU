#!/usr/bin/env python3

import os
import re

import pandas as pd
from datetime import datetime, timedelta

LOGS_DIR = "./raw-logs"
PARSED_LOGS_FILE_NAME_PREFIX = "parsed-logs"

def parse_model(header_line: str):
    if header_line.startswith('#HEADER'):
        model_file = header_line.split(',')[0]
        return model_file[model_file.rfind('/')+1:]
    else:
        return None

def parse_start_dt(log_name: str):
    return datetime.strptime(log_name[0:19], "%Y_%m_%d_%H_%M_%S")

COLUMNS = ['start_dt', 'end_dt', 'benchmark', '#SDC', '#DUE', '#abort', '#end', 'Acc Err', 'Acc Time']
MODEL_TO_BENCH_NAME_MAP = {
    'conv_2d_1_256_256_1_40_40_1_1_quant_edgetpu.tflite': 'Conv 256',
    'conv_2d_1_1024_1024_1_40_40_1_1_quant_edgetpu.tflite': 'Conv 1024',
    'depthwise_conv_2d_1_256_256_3_20_20_3_1_quant_edgetpu.tflite': 'DepthConv 256',
    'depthwise_conv_2d_1_1024_1024_3_20_20_3_1_quant_edgetpu.tflite': 'DepthConv 1024',
}

runs = pd.DataFrame(columns=COLUMNS)

corrupted_logs = {
    'No Header': 0,
    'Zero Iterations': 0,
    'Other': 0
}

for log_filename in os.listdir(LOGS_DIR):
    if not log_filename.endswith('.log'):
        continue

    log_file = open(f"{LOGS_DIR}/{log_filename}")
    lines = log_file.readlines()

    if not lines[0].startswith('#HEADER'):
        corrupted_logs['No Header'] += 1
        continue

    start_dt = parse_start_dt(log_filename)
    model = parse_model(lines[0])
    nsdcs = 0
    ndue = 0
    nabort = 0
    nend = 0

    niterations = 0
    acc_time_line = None
    acc_err_line = None

    for line in lines:
        if line.startswith('#SDC'):
            nsdcs += 1
            acc_time_line = line
            acc_err_line = line
        elif line.startswith('#IT'):
            niterations += 1
            acc_time_line = line
        elif line.startswith('#DUE'):
            ndue += 1
        elif line.startswith('#ABORT'):
            nabort +=1
        elif line.startswith('#END'):
            nend +=1

    if niterations == 0:
        corrupted_logs['Zero Iterations'] += 1
        continue

    if not acc_time_line:
        corrupted_logs['Other'] += 1
        continue

    if not acc_err_line:
        if nsdcs > 0:
            corrupted_logs['Other'] += 1
        continue

    acc_time = float(re.compile('AccTime:(\d+\.\d+)').findall(acc_time_line)[0])
    acc_err = float(re.compile('AccErr:(\d+)').findall(acc_err_line)[0])
    end_dt = start_dt + timedelta(seconds=acc_time)
    benchmark = MODEL_TO_BENCH_NAME_MAP[model]

    run_dict = { c: v for (c, v) in zip(COLUMNS, [start_dt, end_dt, benchmark, nsdcs, ndue, nabort, nend, acc_err, acc_time]) }
    runs = runs.append(run_dict, ignore_index=True)

runs = runs.sort_values('benchmark')
print("> ALL RUNS")
print(runs)

print("> GROUPED BY BENCHMARK")
runs_grouped_by_bench = runs.groupby('benchmark').agg({ c: 'sum' for c in COLUMNS[3:] })
runs_grouped_by_bench['Error Rate'] = runs_grouped_by_bench['#SDC'] / runs_grouped_by_bench['Acc Time']
print(runs_grouped_by_bench)

runs.to_csv(PARSED_LOGS_FILE_NAME_PREFIX + '.csv')
runs_grouped_by_bench.to_csv(PARSED_LOGS_FILE_NAME_PREFIX + 'benchmarks-error-rate.csv')