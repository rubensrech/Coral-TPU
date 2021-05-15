import time
import sys

from log_helper_swig_wraper import log_helper as lh

benchmarkName = "my_benchmark"
benchmarkDetails = "benchmark details, benchmark details, benchmark details, benchmark details"
if lh.start_log_file(benchmarkName, benchmarkDetails) != 0:
    print("Could not start log file")

lh.set_max_errors_iter(32)

print(f"log file is {lh.get_log_file_name()}")

for i in range(40):
    lh.start_iteration()
    time.sleep(1)
    lh.end_iteration()

    err_count = 0
    info_count = 0
    
    if i % 8 == 0:
        lh.log_error_detail("detail of error x")
        error_count = i+1

    if i % 16 == 0:
        lh.log_info_detail("info of event during iteration")
        info_count = info_count + 520

    lh.log_error_count(error_count)
    lh.log_info_count(info_count)

lh.end_log_file()