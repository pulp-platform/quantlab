#!/bin/bash

PROBLEM=${1}

CFG_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
HARD_STORAGE=${CFG_DIR}/hard_storage.json
QUANT_HOME=$(cd $(dirname ${CFG_DIR}) && pwd)

# data folder
PYTHON_READ_HARD_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_HOME_DATA=$(python -c "${PYTHON_READ_HARD_STORAGE_DATA}")/$(basename ${QUANT_HOME})
HARD_DIR_DATA=${HARD_HOME_DATA}/problems/${PROBLEM}/data
mkdir -p ${HARD_DIR_DATA}
echo "QuantLab: remember to import and prepare data for problem ${PROBLEM} at ${HARD_DIR_DATA} ."

# logs folder
PYTHON_READ_HARD_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_HOME_LOGS=$(python -c "${PYTHON_READ_HARD_STORAGE_LOGS}")/$(basename ${QUANT_HOME})
HARD_DIR_LOGS=${HARD_HOME_LOGS}/problems/${PROBLEM}/logs
mkdir -p ${HARD_DIR_LOGS}

# set up problem package
DIR_PROBLEM=${QUANT_HOME}/problems/${PROBLEM}
mkdir ${DIR_PROBLEM}
touch ${DIR_PROBLEM}/config.json
touch ${DIR_PROBLEM}/__init__.py
# data folder
SOFT_DIR_DATA=${DIR_PROBLEM}/data
ln -s ${HARD_DIR_DATA} ${SOFT_DIR_DATA}
# logs folder
SOFT_DIR_LOGS=${DIR_PROBLEM}/logs
ln -s ${HARD_DIR_LOGS} ${SOFT_DIR_LOGS}
