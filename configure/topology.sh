#!/usr/bin/env bash

PROBLEM=${1}   # TODO: check that this is alphanumeric
TOPOLOGY=${2}  # TODO: check that this is alphanumeric

PATH_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PATH_QUANTLAB_HOME=$(cd $(dirname ${PATH_CFG}) && pwd)

# retrieve (hard) logs folder
PATH_STORAGE_CFG=${PATH_CFG}/storage_cfg.json
PYTHON_READ_STORAGE_CFG_LOGS="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
PATH_HARD_SYSTEMS_PKG_4LOGS=$(python -c "${PYTHON_READ_STORAGE_CFG_LOGS}")/$(basename ${PATH_QUANTLAB_HOME})/systems

if [ ! -d "${PATH_HARD_SYSTEMS_PKG_4LOGS}" ]
then
    echo "[QuantLab] QuantLab logs storage is not properly configured!"
else

    PATH_PROBLEM_PKG=${PATH_QUANTLAB_HOME}/systems/${PROBLEM}
    if [ ! -d "${PATH_PROBLEM_PKG}" ] || [ ! -f "${PATH_PROBLEM_PKG}/__init__.py" ]
    then
        echo "[QuantLab] Before configuring a topology sub-package, you must configure the problem package for problem ${PROBLEM}."
    else

        # set up topology sub-package
        PATH_TOPOLOGY_PKG=${PATH_QUANTLAB_HOME}/systems/${PROBLEM}/${TOPOLOGY}
        if [ -d "${PATH_TOPOLOGY_PKG}" ] && [ -f "${PATH_TOPOLOGY_PKG}/__init__.py" ]
        then
            echo "[QuantLab] It seems that the QuantLab sub-package for topology ${TOPOLOGY} has already been created under the problem package for problem ${PROBLEM}..."
        else
            mkdir ${PATH_TOPOLOGY_PKG}
            touch ${PATH_TOPOLOGY_PKG}/__init__.py
            mkdir ${PATH_TOPOLOGY_PKG}/preprocess
            touch ${PATH_TOPOLOGY_PKG}/preprocess/__init__.py
            touch ${PATH_TOPOLOGY_PKG}/"$(tr '[:upper:]' '[:lower:]' <<< ${TOPOLOGY}).py"
            mkdir ${PATH_TOPOLOGY_PKG}/postprocess
            touch ${PATH_TOPOLOGY_PKG}/postprocess/__init__.py
            touch ${PATH_TOPOLOGY_PKG}/config.json
        fi

        # create (hard) logs folder
        PATH_HARD_TOPOLOGY_PKG_LOGS=${PATH_HARD_SYSTEMS_PKG_4LOGS}/${PROBLEM}/${TOPOLOGY}/logs
        mkdir -p ${PATH_HARD_TOPOLOGY_PKG_LOGS}

        # link (soft) logs folder to (hard) logs folder
        ln -s ${PATH_HARD_TOPOLOGY_PKG_LOGS} ${PATH_TOPOLOGY_PKG}/logs

    fi

fi
