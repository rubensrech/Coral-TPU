#!/bin/bash

JSON_FILES=$@
FULL_PATH=$(pwd)
JSON_PARAM="${FULL_PATH}/json_param"

if [ "$#" -eq 0 ];
then
    echo "Usage: ${0} <json-files>"
    exit -1
fi

rm -f ${JSON_PARAM}

for json_file in $JSON_FILES
do
    echo "${FULL_PATH}/${json_file}" >> ${JSON_PARAM}
done