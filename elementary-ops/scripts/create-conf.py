import configparser
import sys
import os

conf_file = "/etc/radiation-benchmarks.conf"
var_dir = "/var/radiation-benchmarks"
log_dir = var_dir + "/log"

# signal_cmd = "killall -q -USR1 killtestSignal.py;"
# signal_cmd += " killall -q -USR1 test_killtest_commands_json.py; killall -q -USR1 python"
# signal_cmd = "killall -q -USR1 killtestSignal-2.0.py;"
# signal_cmd += " killall -q -USR1 test_killtest_commands_json-2.0.py; killall -q -USR1 python3"
signal_cmd = ""

config = configparser.ConfigParser()

config.set("DEFAULT", "vardir", var_dir)
config.set("DEFAULT", "logdir", log_dir)
config.set("DEFAULT", "tmpdir", "/tmp")
config.set("DEFAULT", "signalcmd", signal_cmd)

try:
    if not os.path.isdir(var_dir):
        os.mkdir(var_dir, 0o777)
    os.chmod(var_dir, 0o777)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir, 0o777)
    os.chmod(log_dir, 0o777)
    with open(conf_file, 'w') as configfile:
        config.write(configfile)

except IOError:
    print("I/O Error, please make sure to run as root (sudo)")
    sys.exit(1)