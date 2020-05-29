#!/bin/bash

CFG_DIR=$(cd $(dirname ${BASH_SCRIPT[0]}) && pwd)
HARD_STORAGE_JSON=${CFG_DIR}/hard_storage.json
QUANT_HOME=$(cd $(dirname ${CFG_DIR}) && pwd)

# data folder
PYTHON_READ_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE_JSON}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_STORAGE_DATA_HOME=$(python -c "${PYTHON_READ_STORAGE_DATA}")/$(basename ${QUANT_HOME})
mkdir $HARD_STORAGE_DATA_HOME
mkdir $HARD_STORAGE_DATA_HOME/problems

# logs folder
PYTHON_READ_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE_JSON}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_STORAGE_LOGS_HOME=$(python -c "${PYTHON_READ_STORAGE_LOGS}")/$(basename ${QUANT_HOME})
mkdir $HARD_STORAGE_LOGS_HOME
mkdir $HARD_STORAGE_LOGS_HOME/problems
