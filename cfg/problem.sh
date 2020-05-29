#!/bin/bash

PROBLEM=${1}

CFG_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
HARD_STORAGE_JSON=${CFG_DIR}/hard_storage.json
QUANT_HOME=$(cd $(dirname ${CFG_DIR}) && pwd)

# data folder
PYTHON_READ_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE_JSON}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_STORAGE_DATA_HOME=$(python -c "${PYTHON_READ_STORAGE_DATA}")/$(basename ${QUANT_HOME})
HARD_STORAGE_DATA_HOME_PROBLEM=${HARD_STORAGE_DATA_HOME}/problems/${PROBLEM}
HARD_DIR_DATA=${HARD_STORAGE_DATA_HOME_PROBLEM}/data
mkdir ${HARD_STORAGE_DATA_HOME_PROBLEM}
mkdir ${HARD_DIR_DATA}
echo "QUANTLAB WARNING: import and prepare data for problem ${PROBLEM} at ${HARD_DIR_DATA}"

# logs folder
PYTHON_READ_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE_JSON}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_STORAGE_LOGS_HOME=$(python -c "${PYTHON_READ_STORAGE_LOGS}")/$(basename ${QUANT_HOME})
HARD_STORAGE_LOGS_HOME_PROBLEM=${HARD_STORAGE_LOGS_HOME}/problems/${PROBLEM}
HARD_DIR_LOGS=${HARD_STORAGE_LOGS_HOME_PROBLEM}/logs
mkdir ${HARD_STORAGE_LOGS_HOME_PROBLEM}
mkdir ${HARD_DIR_LOGS}

# set up problem package
HOME_PROBLEM=${QUANT_HOME}/problems/${PROBLEM}
mkdir ${HOME_PROBLEM}
ln -s ${HARD_DIR_DATA} ${HOME_PROBLEM}/data
ln -s ${HARD_DIR_LOGS} ${HOME_PROBLEM}/logs
touch ${HOME_PROBLEM}/config.json
touch ${HOME_PROBLEM}/__init__.py
