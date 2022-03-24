# libLogHelper

This lib is necessary for radiation experiments to log benchmark data. This library is meant to be used within the
benchmark source code.

# Dependencies

- CMake >=3.16
- GCC with std=C11
- SWIG for Python applications

# Getting started

To use with C or C++, you have to build the library and then link with your code. **The library functions are not thread
safe**.

## Building libLogHelper

You can set some library functionalities on CMake:

- -DLOGGING_TYPE=<logging approach> To set the logging approach, that can be LOCAL, UDP, LOCAL_AND_UPD (default is
  LOCAL).
- -DWITH_PYTHON_SUPPORT=OFF to disable Python 3.8 wrapper building
- -DWITH_DEBUG=OFF to disable debug printing information
- -DRAD_BENCHS_INSTALL_DIR=<path to rad benchmarks> (default /home/carol/radiation-benchmarks)
- -DWATCHDOG_COMMANDS=<signal command to be sent to the SW watchdog>
  (default killall -q -USR1 killtestSignal-2.0.py; test_killtest_commands_json-2.0.py;; killall -q -USR1 killall -q
  -USR1 python3;)
- -DTMP_DIR=<System tmp dir> (default /tmp)
- -DECC_INFO_FILE_DIR=<Path to file that will contain 1/0 that refers to ECC enabled or disabled respectively.
- default (/tmp/ecc-info-file.txt)>
- -DSERVER_IP=<Server that will receive the messages IP (default 192.168.1.5)>
- -DSERVER_PORT=<server port that will receive the messages (default 1024)

```shell
cd libLogHelper
mkdir build && cd build
# To build the whole lib
cmake ..
make
```

If you wish to install in the whole system

```shell
sudo make install
```

To uninstall the library (/var/radiation-benchmarks/ path is not deleted)

```shell
sudo make uninstall
```

Then to use you just have to build the benchmark with this library use -lLogHelper with -I<install_path>/include -L<
install_path>/lib
(if not installed in the system)

```C
// include the header in your C/C++ code
#include
"log_helper.h"
...
```

### Python Wrapper

It is possible to use the Python 3.8 wrapper to use the library on Python apps. You have to set the path to the build
folder, then just call the same name as:

```python
import log_helper as lh

lh.start_log_file("MyBenchmark", "Myheader")
...
```