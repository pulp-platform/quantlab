#!/usr/bin/env bash

PROBLEM=${1} # TODO: check that this is alphanumeric

PATH_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PATH_QUANTLAB_HOME=$(cd $(dirname ${PATH_CFG}) && pwd)

# set up problem sub-package
PATH_PROBLEM_PKG=${PATH_QUANTLAB_HOME}/systems/${PROBLEM}
mkdir ${PATH_PROBLEM_PKG}
touch ${PATH_PROBLEM_PKG}/__init__.py
touch ${PATH_PROBLEM_PKG}/meter.py

# retrieve (hard) data folder
PATH_STORAGE_CFG=${PATH_CFG}/storage_cfg.json
PYTHON_READ_STORAGE_CFG_DATA="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
PATH_HARD_PROBLEM_PKG=$(python -c "${PYTHON_READ_STORAGE_CFG_DATA}")/$(basename ${PATH_QUANTLAB_HOME})/systems/${PROBLEM}
mkdir -p ${PATH_HARD_PROBLEM_PKG}/data
echo "[QuantLab message] Remember to import and prepare the data for problem ${PROBLEM} at $(dirname ${PATH_HARD_PROBLEM_PKG}/data) ."

# create (soft) data folder
ln -s ${PATH_HARD_PROBLEM_PKG}/data ${PATH_PROBLEM_PKG}/data
