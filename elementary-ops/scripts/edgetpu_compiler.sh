#!/bin/bash

ARGS=$@

if command -v edgetpu_compiler &> /dev/null; then
    edgetpu_compiler $ARGS
    exit $?
elif command -v docker &> /dev/null; then
    docker run -it --rm -v $(pwd):/home/edgetpu edgetpu_compiler edgetpu_compiler $ARGS
    exit $?
else
    echo "Unsupported"
    exit -1
fi