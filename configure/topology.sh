#!/bin/bash

PROBLEM=${1}   # TODO: check that this is alphanumeric
TOPOLOGY=${2}  # TODO: check that this is alphanumeric

PATH_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PATH_QUANTLAB_HOME=$(cd $(dirname ${PATH_CFG}) && pwd)

# set up topology sub-package
PATH_TOPOLOGY_PKG=${PATH_QUANTLAB_HOME}/systems/${PROBLEM}/${TOPOLOGY}
mkdir ${PATH_TOPOLOGY_PKG}
touch ${PATH_TOPOLOGY_PKG}/__init__.py
mkdir ${PATH_TOPOLOGY_PKG}/preprocess
touch ${PATH_TOPOLOGY_PKG}/preprocess/__init__.py
touch ${PATH_TOPOLOGY_PKG}/"${TOPOLOGY,,}.py"
mkdir ${PATH_TOPOLOGY_PKG}/postprocess
touch ${PATH_TOPOLOGY_PKG}/postprocess/__init__.py
touch ${PATH_TOPOLOGY_PKG}/config.json

# retrieve (hard) logs folder
PATH_STORAGE_CFG=${PATH_CFG}/storage_cfg.json
PYTHON_READ_STORAGE_CFG_LOGS="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
PATH_HARD_TOPOLOGY_PKG=$(python -c "${PYTHON_READ_STORAGE_CFG_LOGS}")/$(basename ${PATH_QUANTLAB_HOME})/systems/${PROBLEM}/${TOPOLOGY}
mkdir -p ${PATH_HARD_TOPOLOGY_PKG}/logs

# create (soft) logs folder
ln -s ${PATH_HARD_TOPOLOGY_PKG}/logs ${PATH_TOPOLOGY_PKG}/logs
