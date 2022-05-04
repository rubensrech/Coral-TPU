#!/usr/bin/env python3

import os
import re
import logging
import argparse
import pandas as pd

from datetime import datetime, timedelta

COLUMNS = ["start_dt", "end_dt", "benchmark", "#SDC", "#DUE", "#abort", "#end", "Acc Err", "Acc Time"]

MODEL_TO_BENCH_NAME_MAP = {
    # Convolutions
    "conv_2d_1_256_256_1_40_40_1_1_quant_edgetpu.tflite": "Conv 256",
    "conv_2d_1_500_500_1_40_40_1_1_quant_edgetpu.tflite": "Conv 500",
    "conv_2d_1_512_512_1_40_40_1_1_quant_edgetpu.tflite": "Conv 512",
    "conv_2d_1_1000_1000_1_40_40_1_1_quant_edgetpu.tflite": "Conv 1000",
    "conv_2d_1_1024_1024_1_40_40_1_1_quant_edgetpu.tflite": "Conv 1024",
    "conv_2d_1_1250_1250_1_40_40_1_1_quant_edgetpu.tflite": "Conv 1250",
    "depthwise_conv_2d_1_256_256_3_20_20_3_1_quant_edgetpu.tflite": "DepthConv 256",
    "depthwise_conv_2d_1_500_500_3_20_20_3_1_quant_edgetpu.tflite": "DepthConv 500",
    "depthwise_conv_2d_1_512_512_3_20_20_3_1_quant_edgetpu.tflite": "DepthConv 512",
    "depthwise_conv_2d_1_1000_1000_3_20_20_3_1_quant_edgetpu.tflite": "DepthConv 1000",
    "depthwise_conv_2d_1_1024_1024_3_20_20_3_1_quant_edgetpu.tflite": "DepthConv 1024",
    # Neural Networks
    "efficientdet_lite3_512_ptq_edgetpu.tflite": "EfficientDet-Lite3",
    "efficientnet-edgetpu-L_quant_edgetpu.tflite": "EfficientNet-L",
    "efficientnet-edgetpu-M_quant_edgetpu.tflite": "EfficientNet-M",
    "efficientnet-edgetpu-S_quant_edgetpu.tflite": "EfficientNet-S",
    "inception_v3_299_quant_edgetpu.tflite": "Inception v3",
    "inception_v4_299_quant_edgetpu.tflite": "Inception v4",
    "ssd_mobilenet_v2_catsdogs_quant_edgetpu.tflite": "SSD MobileNet v2 - Pets",
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite": "SSD MobileNet v2 - COCO",
    "ssd_mobilenet_v2_subcoco14_quant_edgetpu.tflite": "SSD MobileNet v2 - Sub-COCO",
    "ssd_mobilenet_v2_subcoco14_transf_learn_quant_edgetpu.tflite": "SSD MobileNet v2 - Sub-COCO - TL",
    "ssd_mobilenet_v2_transf_learn_catsdogs_quant_edgetpu.tflite": "SSD MobileNet v2 - Pets - TL",
    "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite": "SSD MobileDet - COCO",
    "tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite": "ResNet-50",
    
}

ACC_TIME_PATTERN = re.compile("AccTime:(\d+\.\d+)")
ACC_ERR_PATTERN = re.compile("AccErr:(\d+)")

class CorruptedLogException(Exception): pass

class InvalidIterationLine(CorruptedLogException): pass
class NoHeaderException(CorruptedLogException): pass
class ZeroIterationsException(CorruptedLogException): pass

def parse_model(header_line: str):
    if header_line.startswith("#HEADER"):
        model_file = header_line.split(",")[0]
        return model_file[model_file.rfind("/")+1:]
    else:
        return None

def parse_start_dt(log_name: str):
    return datetime.strptime(log_name[0:19], "%Y_%m_%d_%H_%M_%S")

def find_header_line(lines, max_lines):
    for i in range(min(len(lines), max_lines)):
        if lines[i].startswith("#HEADER"):
            return lines[i]
    return None

def parse_log_file(log_file_path: str):
    log_file = open(log_file_path)
    lines = log_file.readlines()

    log_filename = os.path.basename(log_file_path)
    start_dt = parse_start_dt(log_filename)

    header_line = find_header_line(lines, 10)

    if not header_line:
        raise NoHeaderException()

    model = parse_model(header_line)

    nsdcs = 0
    ndue = 0
    nabort = 0
    nend = 0
    niterations = 0
    acc_time_line = None
    acc_err_line = None

    for line in lines:
        if line.startswith("#SDC"):
            nsdcs += 1
            acc_time_line = line
            acc_err_line = line
        elif line.startswith("#IT"):
            niterations += 1
            acc_time_line = line
        elif line.startswith("#DUE"):
            ndue += 1
        elif line.startswith("#ABORT"):
            nabort +=1
        elif line.startswith("#END"):
            nend +=1

    if niterations == 0:
        raise ZeroIterationsException()
    
    if nend == 0:
        ndue += 1

    if not acc_time_line:
        raise Exception("Sanity check failed")

    if not acc_err_line:
        if nsdcs > 0:
            raise CorruptedLogException()

    acc_time_matches = ACC_TIME_PATTERN.findall(acc_time_line)

    if len(acc_time_matches) == 0:
        raise InvalidIterationLine()

    acc_time = float(acc_time_matches[0])
    acc_err = float(ACC_ERR_PATTERN.findall(acc_err_line)[0]) if acc_err_line is not None else 0
    end_dt = start_dt + timedelta(seconds=acc_time)
    benchmark = MODEL_TO_BENCH_NAME_MAP[model]

    run_data = { c: v for (c, v) in zip(COLUMNS, [start_dt, end_dt, benchmark, nsdcs, ndue, nabort, nend, acc_err, acc_time]) }

    return run_data


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("logs_dir", help="Path to directory containing the log files to be parsed")
    parser.add_argument("--out_file_all_runs", help="Path to output file containing all runs sorted by benchmark name", default=None)
    parser.add_argument("--out_file_benchmarks", help="Path to output file containing all data grouped by benchmark", default=None)
    parser.add_argument("--constant_flux", help='''If provided, the cross sections will be calculated considering
                                                    the flux was constant during the whole execution time (example: 5.18e6)''', type=float, default=None)
    args = parser.parse_args()

    all_runs = pd.DataFrame(columns=COLUMNS)

    corrupted_logs = {
        "No Header": 0,
        "Zero Iterations": 0,
        "Invalid Acc Time": 0,
        "Other": 0
    }

    for curr_path, sub_dirs, files in os.walk(args.logs_dir):
        for file in files:
            file_path = os.path.join(curr_path, file)

            if not file_path.endswith(".log"):
                logging.info(f"Skipping file [{file_path}]")
                continue

            try:
                run_data = parse_log_file(file_path)
                all_runs = all_runs.append(run_data, ignore_index=True)
            except ZeroIterationsException:
                # print(f"ZeroIterationsException: {file}")
                corrupted_logs["Zero Iterations"] += 1
            except NoHeaderException:
                print(f"NoHeaderException: {file}")
                corrupted_logs["No Header"] += 1
            except InvalidIterationLine:
                print(f"InvalidIterationLine: {file}")
                corrupted_logs["Invalid Acc Time"] += 1
            except CorruptedLogException:
                print(f"CorruptedLogException: {file}")
                corrupted_logs["Other"] += 1

    all_runs = all_runs.sort_values("benchmark")

    print("> CORRUPTED LOGS")
    print(corrupted_logs)
    print("\n")

    print("> ALL RUNS")
    print(all_runs)
    print("\n")

    print("> GROUPED BY BENCHMARK")
    runs_grouped_by_bench = all_runs.groupby("benchmark").agg({ c: "sum" for c in COLUMNS[3:] })
    runs_grouped_by_bench["Error Rate"] = runs_grouped_by_bench["#SDC"] / runs_grouped_by_bench["Acc Time"]

    if args.constant_flux is not None:
        # Calculate cross section
        runs_grouped_by_bench["Fluency"] = runs_grouped_by_bench["Acc Time"] * args.constant_flux
        runs_grouped_by_bench["Cross Section SDC"] = runs_grouped_by_bench["#SDC"] / runs_grouped_by_bench["Fluency"]
        runs_grouped_by_bench["Cross Section DUE"] = runs_grouped_by_bench["#DUE"] / runs_grouped_by_bench["Fluency"]

    runs_grouped_by_bench = runs_grouped_by_bench.sort_values("Acc Time", ascending=False)
    print(runs_grouped_by_bench)
    print("\n")

    if args.out_file_all_runs is not None:
        all_runs.to_csv(args.out_file_all_runs)

    if args.out_file_benchmarks is not None:
        runs_grouped_by_bench.to_csv(args.out_file_benchmarks)

main()