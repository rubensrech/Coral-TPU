import sys

# from include.log_helper_swig_wraper import _log_helper as lh

sys.path.insert(0, './include/log_helper_swig_wraper/')
import _log_helper as lh

print(lh.start_log_file("my_benchmark", "benchmark details, benchmark details, benchmark details, benchmark details"))
