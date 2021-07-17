#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
from typing import List
from enum import Enum
from collections import Counter

class ConvSDCErrorsGeometry(Enum):
    UNDEFINED, SINGLE, LINE, SQUARE, RANDOM = "UNDEFINED", "SINGLE", "LINE", "SQUARE", "RANDOM"

class ConvSDCError:
    def __init__(self, pos: List[int], result: int, expected: int):
        self.position = pos
        self.result = result
        self.expected = expected
    
    def __repr__(self):
        return f'{self.__class__.__name__}(pos={self.position},res={self.result},exp={self.expected})'

class ConvSDC:
    def __init__(self, errors: List[ConvSDCError]):
        self.errors = errors

        self.errors_geometry = self.compute_errors_geometry()
        self.errors_magnitude_histogram = self.compute_errors_magnitude_histogram()
        
    def compute_errors_geometry(self):
        corrupted_positions = np.array(list(map(lambda e: e.position, self.errors)))
        
        if len(corrupted_positions) == 0:
            error_format = ConvSDCErrorsGeometry.UNDEFINED
        if len(corrupted_positions) == 1:
            error_format = ConvSDCErrorsGeometry.SINGLE
        elif len(corrupted_positions):
            # Get X dimension of all corrupted positions
            all_x_positions = corrupted_positions[:, 0]

            # Get Y dimension of all corrupted positions
            all_y_positions = corrupted_positions[:, 1]

            # Count how many times each value is in the list
            unique_x_elements, count_x_positions = np.unique(all_x_positions, return_counts=True)
            unique_y_elements, count_y_positions = np.unique(all_y_positions, return_counts=True)

            # Check if any value is in the list more than once
            row_error = np.any(count_x_positions > 1)
            col_error = np.any(count_y_positions > 1)

            if row_error and col_error:  # square error
                error_format = ConvSDCErrorsGeometry.SQUARE
            elif row_error or col_error:  # row/col error
                error_format = ConvSDCErrorsGeometry.LINE
            else:  # random error
                error_format = ConvSDCErrorsGeometry.RANDOM
        return error_format

    def compute_errors_magnitude_histogram(self):
        if len(self.errors) == 0:
            return {}

        res_exp_arr = np.array(list(map(lambda e: [e.result, e.expected], self.errors)))
        res_arr = res_exp_arr[:, 0]
        exp_arr = res_exp_arr[:, 1]
        diff_arr = np.abs(res_arr - exp_arr)
        magnitudes, counts = np.unique(diff_arr, return_counts=True)
        return { mag: count for (mag, count) in zip(magnitudes, counts) }

    def __repr__(self):
        return f'{self.__class__.__name__}(errs={len(self.errors)},geometry={self.errors_geometry},mag_hist={self.errors_magnitude_histogram})'

class ConvLog:
    def __init__(self, log_name: str, model: str, sdcs: List[ConvSDC]):
        self.log_name = log_name
        self.model = model
        self.sdcs = sdcs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(log_name={self.log_name},model={self.model}),sdcs={self.sdcs})'

def is_conv_log(log_filename: str):
    return "Conv2d" in log_filename

def parse_model(header_line: str):
    if header_line.startswith('#HEADER'):
        model_file = header_line.split(',')[0]
        return model_file[model_file.rfind('/')+1:]
    else:
        return None

def parse_conv_err_line(err_line: str):
    pattern = re.compile('#ERR position: \[(\d+(, \d+)+)\], result: (\d+), expected: (\d+)')
    match = pattern.match(err_line)

    if match:
        position_str, result_str, expected_str = match.group(1), match.group(3), match.group(4)
        position = list(map(lambda p: int(p.strip()), position_str.split(',')))
        return ConvSDCError(position, int(result_str), int(expected_str))
    else:
        return None

def parse_conv_logs(logs_dir: str):
    log_files = [os.path.join(logs_dir, log_filename) for log_filename in os.listdir(logs_dir)]
    log_files = list(filter(lambda log: log.endswith('.log') and is_conv_log(log), log_files))

    parsed_logs: List[ConvLog] = []

    for log_filename in log_files:
        log_file = open(log_filename, 'r')

        header_line = None
        curr_log_sdcs = []
        curr_sdc_errors = []

        for line in log_file:
            if line.startswith('#HEADER'):
                header_line = line
            elif line.startswith('#ERR position'):
                sdc_err = parse_conv_err_line(line)
                curr_sdc_errors.append(sdc_err)
            elif line.startswith('#SDC'):
                sdc = ConvSDC(curr_sdc_errors)
                curr_log_sdcs.append(sdc)
                curr_sdc_errors = []

        log_file.close()

        if not header_line:
            print("Corrupted log: no header")
            continue
        
        model = parse_model(header_line)
        parsed_logs.append(ConvLog(log_filename, model, curr_log_sdcs))

    return parsed_logs

def print_stdout_and_file(string, file, indent_level=0):
    content = '  ' * indent_level + string
    print(content)
    print(content, file=file)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('logs_dir', help='Path to directory containing the log files to be parsed')
    parser.add_argument('out_file', help='Path to output file')
    args = parser.parse_args()

    parsed_logs = parse_conv_logs(args.logs_dir)

    models_stats = {}

    for log in parsed_logs:   
        if log.model not in models_stats:
            models_stats[log.model] = {
                'errors_geometry_histogram': Counter({ geometry.value: 0 for geometry in ConvSDCErrorsGeometry }),
                'errors_magnitude_histogram': Counter({}),
            }

        for sdc in log.sdcs:
            models_stats[log.model]['errors_geometry_histogram'][sdc.errors_geometry.value] += 1
            models_stats[log.model]['errors_magnitude_histogram'] += Counter(sdc.errors_magnitude_histogram)

    out_file = open(args.out_file, 'w')

    print_stdout_and_file("---- ERRORS GEOMETRY ANALYSIS FOR CONVOLUTION BENCHMARKS ----\n", out_file)

    for model, stats in models_stats.items():
        print_stdout_and_file(model, out_file)
        print_stdout_and_file("- Errors geometry histogram", out_file, indent_level=1)
        for (geometry, count) in stats['errors_geometry_histogram'].most_common():
            print_stdout_and_file(f"* {geometry}: {count}", out_file, indent_level=2)
        print_stdout_and_file("- Errors magnitude histogram", out_file, indent_level=1)
        if len(stats['errors_magnitude_histogram']) > 0:
            for (mag, count) in stats['errors_magnitude_histogram'].most_common():
                print_stdout_and_file(f"* {mag}: {count}", out_file, indent_level=2)
        else:
            print_stdout_and_file("* No errors", out_file, indent_level=2)
        print_stdout_and_file("", out_file)

if __name__ == "__main__":
    main()