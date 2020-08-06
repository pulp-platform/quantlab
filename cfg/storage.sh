#!/bin/bash

DIR_CFG=$(cd $(dirname ${BASH_SCRIPT[0]}) && pwd)
HARD_STORAGE_CFG=${DIR_CFG}/hard_storage.json
QUANTLAB_HOME=$(cd $(dirname ${DIR_CFG}) && pwd)

# data folder
PYTHON_READ_HARD_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_QUANTLAB_HOME_DATA=$(python -c "${PYTHON_READ_HARD_STORAGE_DATA}")/$(basename ${QUANTLAB_HOME})
mkdir -p ${HARD_QUANTLAB_HOME_DATA}

# logs folder
PYTHON_READ_HARD_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_QUANTLAB_HOME_LOGS=$(python -c "${PYTHON_READ_HARD_STORAGE_LOGS}")/$(basename ${QUANTLAB_HOME})
mkdir -p ${HARD_QUANTLAB_HOME_LOGS}
