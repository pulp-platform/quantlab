#!/usr/bin/env bash

PATH_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PATH_QUANTLAB_HOME=$(cd $(dirname ${PATH_CFG}) && pwd)
PATH_STORAGE_CFG=${PATH_CFG}/storage_cfg.json

# create systems sub-package (for data)
PYTHON_READ_STORAGE_CFG_DATA="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
PATH_HARD_SYSTEMS_PKG_4DATA=$(python -c "${PYTHON_READ_STORAGE_CFG_DATA}")/$(basename ${PATH_QUANTLAB_HOME})/systems
mkdir -p ${PATH_HARD_SYSTEMS_PKG_4DATA}
echo "[QuantLab] QuantLab data sets will be stored in tree under ${PATH_HARD_SYSTEMS_PKG_4DATA}."

# create hard systems sub-package (for logs)
PYTHON_READ_STORAGE_CFG_LOGS="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
PATH_HARD_SYSTEMS_PKG_4LOGS=$(python -c "${PYTHON_READ_STORAGE_CFG_LOGS}")/$(basename ${PATH_QUANTLAB_HOME})/systems
mkdir -p ${PATH_HARD_SYSTEMS_PKG_4LOGS}
echo "[QuantLab] QuantLab logs will be stored in tree under ${PATH_HARD_SYSTEMS_PKG_4LOGS}."
