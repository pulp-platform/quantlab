#!/bin/bash

PROBLEM=${1}

DIR_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
HARD_STORAGE_CFG=${DIR_CFG}/hard_storage.json
QUANTLAB_HOME=$(cd $(dirname ${DIR_CFG}) && pwd)

# (hard) data folder
PYTHON_READ_HARD_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_QUANTLAB_HOME_DATA=$(python -c "${PYTHON_READ_HARD_STORAGE_DATA}")/$(basename ${QUANTLAB_HOME})
HARD_DIR_DATA=${HARD_QUANTLAB_HOME_DATA}/problems/${PROBLEM}/data
mkdir -p ${HARD_DIR_DATA}
echo "QuantLab: remember to import and prepare data for problem ${PROBLEM} at ${HARD_DIR_DATA} ."

# (hard) logs folder
PYTHON_READ_HARD_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_QUANTLAB_HOME_LOGS=$(python -c "${PYTHON_READ_HARD_STORAGE_LOGS}")/$(basename ${QUANTLAB_HOME})
HARD_DIR_LOGS=${HARD_QUANTLAB_HOME_LOGS}/problems/${PROBLEM}/logs
mkdir -p ${HARD_DIR_LOGS}

# set up problem package
DIR_PROBLEM=${QUANTLAB_HOME}/problems/${PROBLEM}
mkdir ${DIR_PROBLEM}
touch ${DIR_PROBLEM}/config.json
touch ${DIR_PROBLEM}/__init__.py

# data folder
DIR_DATA=${DIR_PROBLEM}/data
ln -s ${HARD_DIR_DATA} ${DIR_DATA}

# logs folder
DIR_LOGS=${DIR_PROBLEM}/logs
ln -s ${HARD_DIR_LOGS} ${DIR_LOGS}
